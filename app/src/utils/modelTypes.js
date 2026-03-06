// Model types and constants
import { modelIdToPort, isValidModelId } from './modelIdMapper.js';

/**
 * Model types enumeration
 */
export const MODEL_TYPES = {
  LOCAL: 'local',
  API: 'api'
};

/**
 * Training device types
 */
export const TRAINING_DEVICES = {
  CPU: 'cpu',
  GPU: 'gpu'
};

/**
 * Model status enumeration
 */
export const MODEL_STATUS = {
  IDLE: 'idle',
  LOADING: 'loading',
  ACTIVE: 'active',
  ERROR: 'error',
  TRAINING: 'training',
  PAUSED: 'paused'
};

/**
 * Training status enumeration
 */
export const TRAINING_STATUS = {
  IDLE: 'idle',
  TRAINING: 'training',
  PAUSED: 'paused',
  COMPLETED: 'completed',
  ERROR: 'error'
};

/**
 * Hardware connection status
 */
export const HARDWARE_STATUS = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error'
};

/**
 * Camera status enumeration
 */
export const CAMERA_STATUS = {
  INACTIVE: 'inactive',
  ACTIVE: 'active',
  ERROR: 'error',
  LOADING: 'loading'
};

/**
 * WebSocket connection status
 */
export const WS_STATUS = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  ERROR: 'error'
};

/**
 * Collaboration patterns
 */
export const COLLABORATION_PATTERNS = {
  SEQUENTIAL: 'sequential',
  PARALLEL: 'parallel',
  HYBRID: 'hybrid',
  HIERARCHICAL: 'hierarchical'
};

/**
 * Input types for models
 */
export const INPUT_TYPES = {
  TEXT: 'text',
  IMAGE: 'image',
  AUDIO: 'audio',
  VIDEO: 'video',
  SENSOR: 'sensor',
  MULTIMODAL: 'multimodal'
};

/**
 * Output types for models
 */
export const OUTPUT_TYPES = {
  TEXT: 'text',
  ACTION: 'action',
  DECISION: 'decision',
  VISUALIZATION: 'visualization'
};

/**
 * Priority levels
 */
export const PRIORITY_LEVELS = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical'
};

/**
 * Error types
 */
export const ERROR_TYPES = {
  CONNECTION: 'connection',
  TIMEOUT: 'timeout',
  SERVER: 'server',
  NETWORK: 'network',
  VALIDATION: 'validation',
  CONFIGURATION: 'configuration',
  UNKNOWN: 'unknown'
};

/**
 * Notification types
 */
export const NOTIFICATION_TYPES = {
  SUCCESS: 'success',
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

/**
 * Dataset types
 */
export const DATASET_TYPES = {
  IMAGE: 'image',
  AUDIO: 'audio',
  VIDEO: 'video',
  TEXT: 'text',
  SENSOR: 'sensor',
  MULTIMODAL: 'multimodal'
};

/**
 * Training modes
 */
export const TRAINING_MODES = {
  SUPERVISED: 'supervised',
  UNSUPERVISED: 'unsupervised',
  REINFORCEMENT: 'reinforcement',
  TRANSFER: 'transfer',
  FINE_TUNING: 'fine_tuning'
};

/**
 * Robot training modes
 */
export const ROBOT_TRAINING_MODES = [
  {
    id: 'basic_motion',
    name: 'Basic Motion Training',
    description: 'Train basic joint movements and positions'
  },
  {
    id: 'advanced_motion',
    name: 'Advanced Motion Training',
    description: 'Train complex motion sequences and coordination'
  },
  {
    id: 'vision_guided',
    name: 'Vision Guided Training',
    description: 'Train using camera feedback for visual tasks'
  },
  {
    id: 'sensor_integrated',
    name: 'Sensor Integrated Training',
    description: 'Train using sensor data for balance and stability'
  },
  {
    id: 'task_specific',
    name: 'Task Specific Training',
    description: 'Train for specific tasks like object manipulation'
  }
];

/**
 * Available training models with ports
 */
export const AVAILABLE_TRAINING_MODELS = [
  { id: 'manager', name: 'Manager Model', port: 8001, description: 'Coordinates all other models' },
  { id: 'language', name: 'Language Model', port: 8002, description: 'Natural language processing' },
  { id: 'knowledge', name: 'Knowledge Model', port: 8003, description: 'Knowledge base management' },
  { id: 'vision', name: 'Vision Model', port: 8004, description: 'Visual perception and analysis' },
  { id: 'audio', name: 'Audio Model', port: 8005, description: 'Audio processing and speech recognition' },
  { id: 'autonomous', name: 'Autonomous Model', port: 8006, description: 'Autonomous decision making' },
  { id: 'programming', name: 'Programming Model', port: 8007, description: 'Code generation and analysis' },
  { id: 'planning', name: 'Planning Model', port: 8008, description: 'Task planning and scheduling' },
  { id: 'emotion', name: 'Emotion Model', port: 8009, description: 'Emotion recognition and response' },
  { id: 'spatial', name: 'Spatial Model', port: 8010, description: 'Spatial awareness and navigation' },
  { id: 'computer_vision', name: 'Computer Vision Model', port: 8011, description: 'Advanced computer vision tasks' },
  { id: 'sensor', name: 'Sensor Model', port: 8012, description: 'Sensor data processing and fusion' },
  { id: 'motion', name: 'Motion Model', port: 8013, description: 'Motion control and coordination' },
  { id: 'prediction', name: 'Prediction Model', port: 8014, description: 'Predictive analytics and forecasting' },
  { id: 'advanced_reasoning', name: 'Advanced Reasoning Model', port: 8015, description: 'Complex reasoning tasks' },
  { id: 'data_fusion', name: 'Data Fusion Model', port: 8028, description: 'Multi-source data integration' },
  { id: 'creative_problem_solving', name: 'Creative Problem Solving Model', port: 8017, description: 'Creative solution generation' },
  { id: 'meta_cognition', name: 'Meta Cognition Model', port: 8018, description: 'Self-awareness and reflection' },
  { id: 'value_alignment', name: 'Value Alignment Model', port: 8019, description: 'Ethical decision making' },
  { id: 'vision_image', name: 'Vision Image Model', port: 8020, description: 'Image-specific vision tasks' },
  { id: 'vision_video', name: 'Vision Video Model', port: 8021, description: 'Video-specific vision tasks' },
  { id: 'finance', name: 'Finance Model', port: 8022, description: 'Financial analysis and planning' },
  { id: 'medical', name: 'Medical Model', port: 8023, description: 'Medical knowledge and analysis' },
  { id: 'collaboration', name: 'Collaboration Model', port: 8024, description: 'Team collaboration and coordination' },
  { id: 'optimization', name: 'Optimization Model', port: 8025, description: 'Resource and process optimization' },
  { id: 'computer', name: 'Computer Model', port: 8026, description: 'Computer system knowledge and control' },
  { id: 'mathematics', name: 'Mathematics Model', port: 8027, description: 'Mathematical reasoning and problem solving' }
];

/**
 * Default training parameters
 */
export const DEFAULT_TRAINING_PARAMS = {
  iterations: 1000,
  learningRate: 0.0001,
  batchSize: 32,
  validationSplit: 0.2,
  device: TRAINING_DEVICES.CPU
};

/**
 * Default safety limits for robot training
 */
export const DEFAULT_SAFETY_LIMITS = {
  maxJointVelocity: 5.0,
  maxJointTorque: 10.0,
  maxTemperature: 60,
  emergencyStopThreshold: 1.5
};

/**
 * Collaboration patterns configuration
 */
export const COLLABORATION_PATTERNS_CONFIG = [
  {
    name: 'sequential',
    description: 'Models process data in sequence',
    models: ['manager', 'language', 'knowledge'],
    mode: 'sequential'
  },
  {
    name: 'parallel',
    description: 'Models process data in parallel',
    models: ['vision', 'audio', 'sensor'],
    mode: 'parallel'
  },
  {
    name: 'hierarchical',
    description: 'Manager coordinates specialized models',
    models: ['manager', 'motion', 'vision', 'sensor'],
    mode: 'hierarchical'
  },
  {
    name: 'hybrid',
    description: 'Combination of sequential and parallel processing',
    models: ['manager', 'language', 'vision', 'audio', 'knowledge'],
    mode: 'hybrid'
  }
];

/**
 * API endpoints configuration
 */
export const API_ENDPOINTS = {
  HEALTH: '/health',
  MODELS: '/api/models',
  TRAINING: '/api/training',
  ROBOT_TRAINING: '/api/training',  // Unified training API (supports robot hardware training)
  KNOWLEDGE: '/api/knowledge',
  PROCESS: '/api/process',
  CHAT: '/api/chat',
  DEVICES: '/api/devices',
  CAMERAS: '/api/cameras',
  HARDWARE: '/api/hardware',
  EXTERNAL_API: '/api/external-api'
};

/**
 * WebSocket endpoints
 */
export const WS_ENDPOINTS = {
  VIDEO_STREAM: '/ws/video-stream',
  DEVICE_CONTROL: '/ws/device-control',
  SENSOR_DATA: '/ws/sensor-data',
  TRAINING_STATUS: '/ws/training-status'
};

/**
 * Default configuration values
 */
export const DEFAULT_CONFIG = {
  // Model configuration
  modelRefreshInterval: 5000,
  trainingStatusInterval: 3000,
  
  // WebSocket configuration
  wsReconnectInterval: 5000,
  wsPingInterval: 30000,
  
  // Camera configuration
  defaultCameraResolution: '1280x720',
  defaultCameraFps: 30,
  
  // Training configuration
  defaultTrainingTimeout: 3600000, // 1 hour
  
  // UI configuration
  maxLogItems: 100,
  messageHistoryLimit: 50
};

/**
 * Model port configuration mapping
 */
export const MODEL_PORT_CONFIG = modelIdToPort;

/**
 * Model type definition placeholder
 */
export const Model = {};

/**
 * New model type definition placeholder  
 */
export const NewModel = {};

/**
 * Create a default model configuration
 */
export const createDefaultModel = () => ({
  id: '',
  name: '',
  type: MODEL_TYPES.LOCAL,
  endpoint: '',
  active: false,
  status: MODEL_STATUS.IDLE,
  cpuUsage: 0,
  memoryUsage: 0,
  responseTime: 0,
  port: 8000
});

/**
 * Validate if a port number is valid
 */
export const isValidPort = (port) => {
  return Number.isInteger(port) && port >= 8000 && port <= 9000;
};

/**
 * Check if a model type is an API model
 */
export const isApiModelType = (modelType) => {
  return modelType === MODEL_TYPES.API;
};

// Re-export isValidModelId from modelIdMapper
export { isValidModelId };