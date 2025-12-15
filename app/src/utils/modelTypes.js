// Model types and helper functions

export const MODEL_TYPES = [
  'Manager Model',
  'Language Model',
  'Knowledge Model',
  'Vision Model',
  'Audio Model',
  'Autonomous Model',
  'Programming Model',
  'Planning Model',
  'Emotion Model',
  'Spatial Model',
  'Computer Vision Model',
  'Sensor Model',
  'Motion Model',
  'Prediction Model',
  'Advanced Reasoning Model',
  'Data Fusion Model',
  'Creative Problem Solving Model',
  'Meta Cognition Model',
  'Value Alignment Model',
  'OpenAI API',
  'Anthropic API',
  'Google AI API'
];

export const MODEL_STATUS = [
  'stopped',
  'starting',
  'running',
  'stopping',
  'error'
];

export const MODEL_PORT_CONFIG = {
  'Manager Model': 8001,
  'Language Model': 8002,
  'Knowledge Model': 8003,
  'Vision Model': 8004,
  'Audio Model': 8005,
  'Autonomous Model': 8006,
  'Programming Model': 8007,
  'Planning Model': 8008,
  'Emotion Model': 8009,
  'Spatial Model': 8010,
  'Computer Vision Model': 8011,
  'Sensor Model': 8012,
  'Motion Model': 8013,
  'Prediction Model': 8014,
  'Advanced Reasoning Model': 8015,
  'Data Fusion Model': 8016,
  'Creative Problem Solving Model': 8017,
  'Meta Cognition Model': 8018,
  'Value Alignment Model': 8019
};

// Model class
export class Model {
  constructor(id, name, type) {
    this.id = id;
    this.name = name;
    this.type = type;
    this.description = '';
    this.status = 'stopped';
    this.isActive = false;
    this.isPrimary = false;
    this.port = MODEL_PORT_CONFIG[type] || 0;
    this.lastUpdated = new Date().toISOString();
    this.version = '1.0.0';
    this.apiKey = '';
    this.metrics = null;
  }
}

// New model class for form submission
export class NewModel {
  constructor() {
    this.id = '';
    this.name = '';
    this.type = '';
    this.port = 0;
  }
}

// Create a default model with sensible defaults
export function createDefaultModel(id, name, type) {
  const model = new Model(id, name, type);
  
  // Set default description based on type
  const descriptions = {
    'Manager Model': 'System manager model for coordination',
    'Language Model': 'Natural language processing model',
    'Knowledge Model': 'Knowledge base and retrieval model',
    'Vision Model': 'Computer vision and image processing model',
    'Audio Model': 'Audio processing and speech recognition model',
    'Autonomous Model': 'Self-governing and decision-making model',
    'Programming Model': 'Code generation and software development model',
    'Planning Model': 'Strategic planning and execution model',
    'Emotion Model': 'Emotional analysis and response model',
    'Spatial Model': 'Spatial reasoning and navigation model',
    'Computer Vision Model': 'Advanced computer vision capabilities',
    'Sensor Model': 'Sensor data processing and integration',
    'Motion Model': 'Motion planning and control model',
    'Prediction Model': 'Predictive analytics and forecasting model',
    'Advanced Reasoning Model': 'Complex logical reasoning capabilities',
    'Data Fusion Model': 'Multi-source data integration and fusion',
    'Creative Problem Solving Model': 'Innovative problem-solving approaches',
    'Meta Cognition Model': 'Self-awareness and cognitive monitoring',
    'Value Alignment Model': 'Ethical decision making and value alignment',
    'OpenAI API': 'OpenAI language model integration',
    'Anthropic API': 'Anthropic language model integration',
    'Google AI API': 'Google AI services integration'
  };
  
  model.description = descriptions[type] || '';
  
  return model;
}

// Validate model ID
export function isValidModelId(id) {
  // Model ID should be alphanumeric and can include underscores
  const pattern = /^[a-zA-Z0-9_]+$/;
  return pattern.test(id) && id.length >= 3 && id.length <= 30;
}

// Validate port number
export function isValidPort(port) {
  const portNumber = parseInt(port, 10);
  return !isNaN(portNumber) && portNumber >= 8001 && portNumber <= 8019;
}

// Check if a model type is an API model
export function isApiModelType(type) {
  return type.toLowerCase().includes('api');
}

// Get model type category (local or api)
export function getModelCategory(type) {
  return isApiModelType(type) ? 'api' : 'local';
}

// Get port range info
export function getPortRangeInfo() {
  return {
    min: 8001,
    max: 8019,
    total: 19
  };
}

// Generate mock metrics for a model
export function generateMockMetrics() {
  return {
    memoryUsage: Math.floor(Math.random() * 500) + 100, // 100-600 MB
    cpuUsage: Math.floor(Math.random() * 40) + 5, // 5-45%
    responseTime: Math.floor(Math.random() * 300) + 50, // 50-350 ms
    requestCount: Math.floor(Math.random() * 1000) + 100,
    errorRate: (Math.random() * 0.5).toFixed(2) // 0-0.5%
  };
}