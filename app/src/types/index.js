// AI Brain System Type Definitions

/**
 * Model type definition
 * @typedef {Object} Model
 * @property {string} id - Unique identifier for the model
 * @property {string} name - Display name for the model
 * @property {string} type - Model type (local, api)
 * @property {string} endpoint - API endpoint if applicable
 * @property {string} [apiKey] - API key if applicable (not stored in frontend)
 * @property {string} [modelName] - Name of the model
 * @property {boolean} active - Whether the model is active
 * @property {string} status - Current status (connected, disconnected, running, stopped)
 * @property {number} cpuUsage - CPU usage percentage
 * @property {number} memoryUsage - Memory usage in MB
 * @property {number} responseTime - Response time in ms
 * @property {string} port - Port number the model is running on
 */

export const ModelTypes = {
  LOCAL: 'local',
  API: 'api'
};

export const ModelStatuses = {
  CONNECTED: 'connected',
  DISCONNECTED: 'disconnected',
  RUNNING: 'running',
  STOPPED: 'stopped',
  TESTING: 'testing',
  FAILED: 'failed'
};

export const ModelCategories = {
  MANAGER: 'manager',
  LANGUAGE: 'language',
  KNOWLEDGE: 'knowledge',
  VISION: 'vision',
  AUDIO: 'audio',
  AUTONOMOUS: 'autonomous',
  PROGRAMMING: 'programming',
  PLANNING: 'planning',
  EMOTION: 'emotion',
  SPATIAL: 'spatial',
  COMPUTER_VISION: 'computer_vision',
  SENSOR: 'sensor',
  MOTION: 'motion',
  PREDICTION: 'prediction',
  ADVANCED_REASONING: 'advanced_reasoning',
  DATA_FUSION: 'data_fusion',
  CREATIVE_PROBLEM_SOLVING: 'creative_problem_solving',
  META_COGNITION: 'meta_cognition',
  VALUE_ALIGNMENT: 'value_alignment'
};

/**
 * @returns {Array<Model>} Default mock models for development
 */
export const getDefaultMockModels = () => {
  return [
    {
      id: 'manager',
      name: 'Manager Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 15,
      memoryUsage: 256,
      responseTime: 120,
      port: '8001'
    },
    {
      id: 'language',
      name: 'Language Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 25,
      memoryUsage: 512,
      responseTime: 150,
      port: '8002'
    },
    {
      id: 'knowledge',
      name: 'Knowledge Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 20,
      memoryUsage: 384,
      responseTime: 130,
      port: '8003'
    },
    {
      id: 'vision',
      name: 'Vision Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 30,
      memoryUsage: 768,
      responseTime: 200,
      port: '8004'
    },
    {
      id: 'audio',
      name: 'Audio Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 10,
      memoryUsage: 128,
      responseTime: 80,
      port: '8005'
    },
    {
      id: 'autonomous',
      name: 'Autonomous Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8006'
    },
    {
      id: 'programming',
      name: 'Programming Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 18,
      memoryUsage: 256,
      responseTime: 110,
      port: '8007'
    },
    {
      id: 'planning',
      name: 'Planning Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8008'
    },
    {
      id: 'emotion',
      name: 'Emotion Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8009'
    },
    {
      id: 'spatial',
      name: 'Spatial Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8010'
    },
    {
      id: 'computer_vision',
      name: 'Computer Vision Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 35,
      memoryUsage: 1024,
      responseTime: 250,
      port: '8011'
    },
    {
      id: 'sensor',
      name: 'Sensor Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8012'
    },
    {
      id: 'motion',
      name: 'Motion Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8013'
    },
    {
      id: 'prediction',
      name: 'Prediction Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8014'
    },
    {
      id: 'advanced_reasoning',
      name: 'Advanced Reasoning Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: true,
      status: ModelStatuses.RUNNING,
      cpuUsage: 22,
      memoryUsage: 448,
      responseTime: 180,
      port: '8015'
    },
    {
      id: 'data_fusion',
      name: 'Data Fusion Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8016'
    },
    {
      id: 'creative_problem_solving',
      name: 'Creative Problem Solving Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8017'
    },
    {
      id: 'meta_cognition',
      name: 'Meta Cognition Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8018'
    },
    {
      id: 'value_alignment',
      name: 'Value Alignment Model',
      type: ModelTypes.LOCAL,
      endpoint: '',
      active: false,
      status: ModelStatuses.STOPPED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: '8019'
    },
    {
      id: 'openai',
      name: 'OpenAI Integration',
      type: ModelTypes.API,
      endpoint: 'https://api.openai.com/v1',
      active: false,
      status: ModelStatuses.DISCONNECTED,
      cpuUsage: 0,
      memoryUsage: 0,
      responseTime: 0,
      port: 'external'
    }
  ];
};