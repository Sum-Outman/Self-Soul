/**
 * 统一前端配置管理器
 * 提供前端配置的集中管理，支持环境变量、本地存储和默认值
 * @typedef {Object} ApiConfig
 * @property {string} baseUrl - API基础URL
 * @property {number} timeout - 请求超时时间（毫秒）
 * @property {Object} headers - 请求头
 * 
 * @typedef {Object} SystemConfig
 * @property {number} frontendPort - 前端端口
 * @property {number} backendPort - 后端端口
 * @property {number} websocketPort - WebSocket端口
 * @property {'development'|'staging'|'production'} environment - 环境类型
 * @property {boolean} debug - 调试模式
 * @property {string} logLevel - 日志级别
 * 
 * @typedef {Object} ServiceConfig
 * @property {Object} realtimeStream - 实时流服务配置
 * @property {string} realtimeStream.host - 主机名
 * @property {number} realtimeStream.port - 端口
 * @property {string} realtimeStream.path - 路径
 * @property {Object} valueAlignment - 价值对齐服务配置
 * @property {string} valueAlignment.host - 主机名
 * @property {number} valueAlignment.port - 端口
 * 
 * @typedef {Object} UiConfig
 * @property {Object} notifications - 通知配置
 * @property {number} notifications.defaultDuration - 默认显示时长（毫秒）
 * @property {number} notifications.maxVisible - 最大可见通知数
 * @property {string} notifications.position - 通知位置
 * @property {Object} theme - 主题配置
 * @property {string} theme.primaryColor - 主题色
 * @property {string} theme.secondaryColor - 次要颜色
 * @property {boolean} theme.darkMode - 深色模式
 * 
 * @typedef {Object} ModelsConfig
 * @property {string} defaultLanguageModel - 默认语言模型
 * @property {Object} ollama - Ollama配置
 * @property {string} ollama.baseUrl - Ollama基础URL
 * @property {string} ollama.apiPath - API路径
 * @property {string} ollama.defaultModel - 默认模型
 * 
 * @typedef {Object} AppConfig
 * @property {ApiConfig} api - API配置
 * @property {SystemConfig} system - 系统配置
 * @property {ServiceConfig} services - 服务配置
 * @property {UiConfig} ui - UI配置
 * @property {ModelsConfig} models - 模型配置
 */

// 默认配置
/** @type {AppConfig} */
const DEFAULT_CONFIG = {
  // API配置
  api: {
    baseUrl: '',
    timeout: 120000, // 120秒
    headers: {
      'Content-Type': 'application/json; charset=utf-8'
    }
  },
  
  // 系统配置
  system: {
    frontendPort: 5175,
    backendPort: 8000,
    websocketPort: 8766,
    environment: 'development',
    debug: false,
    logLevel: 'info'
  },
  
  // 服务配置
  services: {
    realtimeStream: {
      host: 'localhost',
      port: 8025,
      path: '/ws'
    },
    valueAlignment: {
      host: 'localhost',
      port: 8019
    }
  },
  
  // UI配置
  ui: {
    notifications: {
      defaultDuration: 5000, // 5秒
      maxVisible: 5,
      position: 'top-right'
    },
    theme: {
      primaryColor: '#2196f3',
      secondaryColor: '#ff9800',
      darkMode: false
    }
  },
  
  // 模型配置
  models: {
    defaultLanguageModel: 'language',
    ollama: {
      baseUrl: 'http://localhost:11434',
      apiPath: '/v1/chat/completions',
      defaultModel: 'llama2'
    }
  }
};

/**
 * 统一配置管理器类
 */
class ConfigManager {
  /**
   * 创建配置管理器实例
   * @constructor
   */
  constructor() {
    /** @type {AppConfig} */
    this.config = { ...DEFAULT_CONFIG };
    this._loadEnvConfig();
    this._loadLocalStorageConfig();
    this._validateConfig();
  }
  
  /**
   * 加载环境变量配置
   * @private
   */
  _loadEnvConfig() {
    // 从Vite环境变量加载配置
    const env = import.meta.env;
    
    if (env) {
      // API配置
      if (env.VITE_API_BASE_URL) {
        this.config.api.baseUrl = env.VITE_API_BASE_URL;
      }
      
      // 系统配置
      if (env.VITE_FRONTEND_PORT) {
        this.config.system.frontendPort = parseInt(env.VITE_FRONTEND_PORT, 10);
      }
      
      if (env.VITE_BACKEND_PORT) {
        this.config.system.backendPort = parseInt(env.VITE_BACKEND_PORT, 10);
      }
      
      if (env.VITE_ENVIRONMENT) {
        this.config.system.environment = env.VITE_ENVIRONMENT;
      }
      
      if (env.VITE_DEBUG) {
        this.config.system.debug = env.VITE_DEBUG === 'true';
      }
      
      if (env.VITE_LOG_LEVEL) {
        this.config.system.logLevel = env.VITE_LOG_LEVEL;
      }
      
      // 服务配置
      if (env.VITE_REALTIME_STREAM_HOST) {
        this.config.services.realtimeStream.host = env.VITE_REALTIME_STREAM_HOST;
      }
      
      if (env.VITE_REALTIME_STREAM_PORT) {
        this.config.services.realtimeStream.port = parseInt(env.VITE_REALTIME_STREAM_PORT, 10);
      }
      
      if (env.VITE_VALUE_ALIGNMENT_HOST) {
        this.config.services.valueAlignment.host = env.VITE_VALUE_ALIGNMENT_HOST;
      }
      
      if (env.VITE_VALUE_ALIGNMENT_PORT) {
        this.config.services.valueAlignment.port = parseInt(env.VITE_VALUE_ALIGNMENT_PORT, 10);
      }
    }
  }
  
  /**
   * 加载本地存储配置
   * @private
   */
  _loadLocalStorageConfig() {
    try {
      const savedConfig = localStorage.getItem('frontend_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        this._mergeConfig(this.config, parsedConfig);
      }
    } catch (error) {
      console.warn('Failed to load config from localStorage:', error);
    }
  }
  
  /**
   * 验证配置
   * @private
   */
  _validateConfig() {
    // 验证端口号
    const validatePort = (port, name) => {
      if (port < 1 || port > 65535) {
        console.warn(`Invalid ${name} port: ${port}, using default`);
        return DEFAULT_CONFIG[name] || 80;
      }
      return port;
    };
    
    this.config.system.frontendPort = validatePort(this.config.system.frontendPort, 'frontendPort');
    this.config.system.backendPort = validatePort(this.config.system.backendPort, 'backendPort');
    this.config.system.websocketPort = validatePort(this.config.system.websocketPort, 'websocketPort');
    this.config.services.realtimeStream.port = validatePort(this.config.services.realtimeStream.port, 'realtimeStreamPort');
    this.config.services.valueAlignment.port = validatePort(this.config.services.valueAlignment.port, 'valueAlignmentPort');
    
    // 验证环境
    const validEnvironments = ['development', 'staging', 'production'];
    if (!validEnvironments.includes(this.config.system.environment)) {
      console.warn(`Invalid environment: ${this.config.system.environment}, using development`);
      this.config.system.environment = 'development';
    }
  }
  
  /**
   * 合并配置（深度合并）
   * @private
   */
  _mergeConfig(target, source) {
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        if (!target[key] || typeof target[key] !== 'object') {
          target[key] = {};
        }
        this._mergeConfig(target[key], source[key]);
      } else {
        target[key] = source[key];
      }
    }
  }
  
  /**
   * 获取配置值
   * @param {string} path - 配置路径，用点号分隔，如 'api.baseUrl'
   * @param {*} defaultValue - 默认值
   * @returns {*} 配置值
   * @template T
   * @returns {T}
   */
  get(path, defaultValue = null) {
    const keys = path.split('.');
    let value = this.config;
    
    for (const key of keys) {
      if (value && typeof value === 'object' && key in value) {
        value = value[key];
      } else {
        return defaultValue;
      }
    }
    
    return value !== undefined ? value : defaultValue;
  }
  
  /**
   * 设置配置值
   * @param {string} path - 配置路径
   * @param {*} value - 配置值
   * @param {boolean} persist - 是否持久化到localStorage
   */
  set(path, value, persist = false) {
    const keys = path.split('.');
    let config = this.config;
    
    // 导航到路径的父级
    for (let i = 0; i < keys.length - 1; i++) {
      const key = keys[i];
      if (!config[key] || typeof config[key] !== 'object') {
        config[key] = {};
      }
      config = config[key];
    }
    
    // 设置值
    const lastKey = keys[keys.length - 1];
    config[lastKey] = value;
    
    // 如果需要持久化，保存到localStorage
    if (persist) {
      this._saveToLocalStorage();
    }
  }
  
  /**
   * 保存配置到localStorage
   * @private
   */
  _saveToLocalStorage() {
    try {
      localStorage.setItem('frontend_config', JSON.stringify(this.config));
    } catch (error) {
      console.error('Failed to save config to localStorage:', error);
    }
  }
  
  /**
   * 获取完整配置
   * @returns {object} 完整配置对象
   */
  getAll() {
    return { ...this.config };
  }
  
  /**
   * 重置配置到默认值
   * @param {boolean} clearPersisted - 是否清除持久化配置
   */
  reset(clearPersisted = false) {
    this.config = { ...DEFAULT_CONFIG };
    this._loadEnvConfig();
    
    if (clearPersisted) {
      localStorage.removeItem('frontend_config');
    } else {
      this._loadLocalStorageConfig();
    }
    
    this._validateConfig();
  }
  
  /**
   * 获取API基础URL
   * @returns {string} API基础URL
   */
  getApiBaseUrl() {
    const baseUrl = this.get('api.baseUrl', '');
    if (baseUrl) {
      return baseUrl;
    }
    
    // 如果没有配置baseUrl，使用相对路径
    return '';
  }
  
  /**
   * 获取后端完整URL
   * @param {string} endpoint - API端点
   * @returns {string} 完整URL
   */
  getBackendUrl(endpoint = '') {
    const baseUrl = this.getApiBaseUrl();
    if (baseUrl) {
      return `${baseUrl}${endpoint}`;
    }
    
    // 如果没有配置baseUrl，使用相对路径
    return endpoint;
  }
  
  /**
   * 获取WebSocket URL
   * @param {string} path - WebSocket路径
   * @returns {string} WebSocket URL
   */
  getWebSocketUrl(path = '/ws') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const hostname = window.location.hostname;
    const port = this.get('system.websocketPort', 8766);
    
    return `${protocol}//${hostname}:${port}${path}`;
  }
  
  /**
   * 获取实时流WebSocket URL
   * @returns {string} 实时流WebSocket URL
   */
  getRealtimeStreamWebSocketUrl() {
    const service = this.get('services.realtimeStream');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    
    return `${protocol}//${service.host}:${service.port}${service.path}`;
  }
  
  /**
   * 获取价值对齐服务URL
   * @returns {string} 价值对齐服务URL
   */
  getValueAlignmentServiceUrl() {
    const service = this.get('services.valueAlignment');
    const protocol = window.location.protocol === 'https:' ? 'https:' : 'http:';
    
    return `${protocol}//${service.host}:${service.port}`;
  }
  
  /**
   * 是否是开发环境
   * @returns {boolean}
   */
  isDevelopment() {
    return this.get('system.environment') === 'development';
  }
  
  /**
   * 是否是生产环境
   * @returns {boolean}
   */
  isProduction() {
    return this.get('system.environment') === 'production';
  }
  
  /**
   * 获取前端URL（用于文档和帮助页面）
   * @returns {string} 前端URL
   */
  getFrontendUrl() {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = this.get('system.frontendPort', 5175);
    
    return `${protocol}//${hostname}:${port}`;
  }
  
  /**
   * 获取后端文档URL
   * @returns {string} 后端文档URL
   */
  getBackendDocsUrl() {
    const protocol = window.location.protocol;
    const hostname = window.location.hostname;
    const port = this.get('system.backendPort', 8000);
    
    return `${protocol}//${hostname}:${port}/docs`;
  }
}

// 创建单例实例
const configManager = new ConfigManager();

// 导出单例和类
export { configManager, ConfigManager };
export default configManager;