/**
 * Minimal error handler utility
 * Simplified to avoid Vue 2 compatibility issues
 */

// Environment detection
const isDevelopment = import.meta.env.MODE === 'development';

// Error log management (simplified)
const errorLogs = [];
const maxLogs = 1000;

/**
 * Add log entry
 */
export function addToLogs(logData) {
  errorLogs.push(logData);
  if (errorLogs.length > maxLogs) {
    errorLogs.shift();
  }
}

/**
 * Handle errors with native console
 */
export function handleError(error, context = 'Unknown', options = {}) {
  const errorObj = error instanceof Error ? error : new Error(String(error));
  
  // Only use native console for error logging
  if (isDevelopment) {
    console.error(`[${context}] Error:`, errorObj.message);
    if (errorObj.stack) {
      console.error(errorObj.stack);
    }
  }
  
  // Add to logs (for internal tracking only)
  const errorData = {
    timestamp: new Date().toISOString(),
    context,
    message: errorObj.message,
    stack: errorObj.stack,
    ...options
  };
  addToLogs(errorData);
  
  return errorData;
}

/**
 * Handle warnings with native console
 */
export function handleWarning(message, context = 'Unknown') {
  if (isDevelopment) {
    console.warn(`[${context}] Warning:`, message);
  }
  
  addToLogs({
    timestamp: new Date().toISOString(),
    context,
    message,
    type: 'warning'
  });
}

/**
 * Log information with native console
 */
export function logInfo(message, context = 'System') {
  if (isDevelopment) {
    console.info(`[${context}] Info:`, message);
  }
  
  addToLogs({
    timestamp: new Date().toISOString(),
    context,
    message,
    type: 'info'
  });
}

/**
 * Get recent error logs
 */
export function getErrorLogs(limit = 100) {
  return errorLogs.slice(-limit);
}

/**
 * Clear all error logs
 */
export function clearErrorLogs() {
  errorLogs.length = 0;
}

// Default export with basic functions
export default {
  handleError,
  handleWarning,
  logInfo,
  getErrorLogs,
  clearErrorLogs
};
