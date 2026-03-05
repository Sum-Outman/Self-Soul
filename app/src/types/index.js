// AI Soul System Type Definitions

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

