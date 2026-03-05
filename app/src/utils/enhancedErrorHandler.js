// Enhanced error handling utilities

/**
 * Enhanced error handler with detailed error categorization and logging
 * @param {Error} error - The error object
 * @param {string} context - Error context for better debugging
 * @returns {Object} Error information object
 */
export const handleEnhancedError = (error, context = 'Unknown context') => {
  console.error(`[Enhanced Error Handler] ${context}`, error);
  
  // Categorize error type
  let errorType = 'unknown';
  let errorMessage = error.message || 'Unknown error';
  let errorDetails = {};
  
  if (error.code === 'ECONNREFUSED') {
    errorType = 'connection';
    errorMessage = 'Backend server is not available. Please start the server.';
  } else if (error.code === 'ECONNABORTED') {
    errorType = 'timeout';
    errorMessage = 'Request timed out. Please check your network connection.';
  } else if (error.response) {
    errorType = 'server';
    errorMessage = error.response.data?.message || error.response.statusText || 'Server error';
    errorDetails = {
      status: error.response.status,
      statusText: error.response.statusText,
      data: error.response.data
    };
  } else if (error.request) {
    errorType = 'network';
    errorMessage = 'No response from server. Please check your connection.';
  }
  
  const errorInfo = {
    type: errorType,
    message: errorMessage,
    details: errorDetails,
    originalError: error,
    context,
    timestamp: new Date().toISOString()
  };
  
  // Log error based on type
  switch (errorType) {
    case 'connection':
      logError(`[${context}] Backend connection failed: ${errorMessage}`);
      break;
    case 'timeout':
      logWarning(`[${context}] Request timed out: ${errorMessage}`);
      break;
    case 'server':
      logError(`[${context}] Server error (${errorDetails.status}): ${errorMessage}`);
      break;
    case 'network':
      logWarning(`[${context}] Network error: ${errorMessage}`);
      break;
    default:
      logError(`[${context}] Unknown error: ${errorMessage}`);
  }
  
  return errorInfo;
};

/**
 * Handle API errors specifically
 * @param {Error} error - The error object
 * @param {string} context - Error context
 * @returns {Object} Processed error object
 */
export const handleApiError = (error, context = 'API call') => {
  console.error(`${context} error:`, error);
  
  // Handle different types of errors
  if (error.response) {
    // Server returned an error status
    return {
      success: false,
      message: error.response.data?.message || `Server error: ${error.response.status}`,
      error: error.response.data?.error || error.response.statusText,
      status: error.response.status
    };
  } else if (error.request) {
    // Request was made but no response
    return {
      success: false,
      message: 'No response from server. Please check your connection.',
      error: 'Network error',
      status: 0
    };
  } else {
    // Other errors
    return {
      success: false,
      message: error.message || 'Unknown error',
      error: error.message || 'Unknown error',
      status: 0
    };
  }
};

/**
 * Handle network errors specifically
 * @param {Error} error - The error object
 * @param {string} serviceName - Name of the service that failed
 * @returns {Object} Processed error object
 */
export const handleNetworkError = (error, serviceName = 'service') => {
  console.error(`Network error with ${serviceName}:`, error);
  
  let message = `Failed to connect to ${serviceName}`;
  
  if (error.code === 'ECONNREFUSED') {
    message = `${serviceName} is not running or unreachable`;
  } else if (error.code === 'ECONNABORTED') {
    message = `Connection to ${serviceName} timed out`;
  }
  
  return {
    success: false,
    message,
    error: error.message || 'Network error',
    code: error.code
  };
};

/**
 * Handle configuration errors
 * @param {Error} error - The error object
 * @param {string} configName - Name of the configuration
 * @returns {Object} Processed error object
 */
export const handleConfigError = (error, configName = 'configuration') => {
  console.error(`${configName} error:`, error);
  
  return {
    success: false,
    message: `Failed to initialize ${configName}`,
    error: error.message || 'Configuration error',
    configName
  };
};

/**
 * Handle validation errors
 * @param {Error} error - The error object
 * @param {string} validationContext - Validation context
 * @returns {Object} Processed error object
 */
export const handleValidationError = (error, validationContext = 'validation') => {
  console.error(`${validationContext} error:`, error);
  
  return {
    success: false,
    message: `Validation failed for ${validationContext}`,
    error: error.message || 'Validation error',
    validationContext
  };
};

/**
 * Log information messages
 * @param {string} message - Information message
 */
export const logInfo = (message) => {
  console.log(`[INFO] ${new Date().toLocaleTimeString()}: ${message}`);
};

/**
 * Log warning messages
 * @param {string} message - Warning message
 */
export const logWarning = (message) => {
  console.warn(`[WARNING] ${new Date().toLocaleTimeString()}: ${message}`);
};

/**
 * Log error messages
 * @param {string} message - Error message
 */
export const logError = (message) => {
  console.error(`[ERROR] ${new Date().toLocaleTimeString()}: ${message}`);
};

/**
 * Log success messages
 * @param {string} message - Success message
 */
export const logSuccess = (message) => {
  console.log(`[SUCCESS] ${new Date().toLocaleTimeString()}: ${message}`);
};

/**
 * Log debug messages (only in development mode)
 * @param {string} message - Debug message
 * @param {Object} data - Additional debug data
 */
export const logDebug = (message, data = {}) => {
  if (import.meta.env.DEV) {
    console.debug(`[DEBUG] ${new Date().toLocaleTimeString()}: ${message}`, data);
  }
};

/**
 * Generic error response handler for API calls
 * @param {Error} error - The error object
 * @param {Function} errorCallback - Optional error callback function
 * @returns {Object} Default error response
 */
export const handleErrorResponse = (error, errorCallback = null) => {
  const errorInfo = handleEnhancedError(error);
  
  if (errorCallback && typeof errorCallback === 'function') {
    errorCallback(errorInfo);
  }
  
  return {
    success: false,
    message: errorInfo.message,
    error: errorInfo
  };
};

/**
 * Safe JSON parser with error handling
 * @param {string} jsonString - JSON string to parse
 * @param {*} defaultValue - Default value if parsing fails
 * @returns {*} Parsed JSON or default value
 */
export const safeJsonParse = (jsonString, defaultValue = null) => {
  try {
    return JSON.parse(jsonString);
  } catch (error) {
    logWarning(`Failed to parse JSON: ${error.message}`);
    return defaultValue;
  }
};

/**
 * Safe local storage operation with error handling
 * @param {string} operation - Operation type: 'get', 'set', 'remove', 'clear'
 * @param {string} key - Storage key
 * @param {*} value - Value for 'set' operation
 * @returns {*} Result of the operation or null on error
 */
export const safeLocalStorage = (operation, key, value = null) => {
  try {
    switch (operation) {
      case 'get':
        return localStorage.getItem(key);
      case 'set':
        localStorage.setItem(key, value);
        return true;
      case 'remove':
        localStorage.removeItem(key);
        return true;
      case 'clear':
        localStorage.clear();
        return true;
      default:
        return null;
    }
  } catch (error) {
    logError(`Local storage ${operation} failed: ${error.message}`);
    return null;
  }
};