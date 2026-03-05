// Error handling utility for API calls and application errors

// Safe utility to stringify any object without throwing errors
const safeStringify = (obj, indent = 2) => {
  try {
    if (obj === undefined) return 'undefined';
    if (obj === null) return 'null';
    
    // Handle primitive types
    if (typeof obj !== 'object') return String(obj);
    
    // Handle Error objects specially
    if (obj instanceof Error) {
      return JSON.stringify({
        name: obj.name,
        message: obj.message,
        stack: obj.stack
      }, null, indent);
    }
    
    // For other objects, try to create a safe representation
    const safeObj = {};
    try {
      // Try to get own property names safely
      const keys = Object.keys(obj);
      for (const key of keys) {
        try {
          // Try to get value - this might throw for some getters
          const value = obj[key];
          // Recursively stringify but with depth limit
          if (value === undefined || value === null) {
            safeObj[key] = value;
          } else if (typeof value === 'function') {
            safeObj[key] = '[Function]';
          } else if (typeof value === 'object') {
            // Limit depth to avoid circular references
            try {
              safeObj[key] = JSON.parse(safeStringify(value, 0));
            } catch {
              safeObj[key] = '[Object]';
            }
          } else {
            safeObj[key] = value;
          }
        } catch {
          safeObj[key] = '[Error accessing property]';
        }
      }
    } catch {
      // If we can't even get keys, return a basic representation
      return `[Object: ${typeof obj}]`;
    }
    
    return JSON.stringify(safeObj, null, indent);
  } catch (error) {
    return `[Could not stringify object: ${error.message || 'Unknown error'}]`;
  }
};

export const handleApiError = (error, context = 'API call') => {
  // Handle case where error is undefined or null
  if (error === undefined || error === null) {
    console.error(`${context} error: undefined error object`);
    return {
      success: false,
      message: 'Unknown error occurred',
      error: 'Error object is undefined',
      status: 0
    };
  }
  
  // Safely log the error with multiple layers of protection
  try {
    // First try to safely stringify the error
    const errorStr = safeStringify(error);
    console.error(`${context} error:`, errorStr);
  } catch (logError) {
    // If stringify fails, try a simpler approach
    try {
      console.error(`${context} error:`, String(error));
    } catch {
      // Last resort
      console.error(`${context} error: [Error object could not be logged]`);
    }
  }
  
  // Handle different types of errors with safe property access
  try {
    // Check if error has response property (Axios error) with safe access
    if (error && typeof error === 'object' && 'response' in error) {
      const response = error.response;
      if (response !== undefined && response !== null) {
        // Server returned an error status
        const data = response.data;
        return {
          success: false,
          message: (data && data.message) || `Server error: ${response.status || 500}`,
          error: (data && data.error) || response.statusText || 'Unknown server error',
          status: response.status || 500
        };
      }
    }
    
    // Check if error has request property (Axios network error)
    if (error && typeof error === 'object' && 'request' in error) {
      // Request was made but no response
      return {
        success: false,
        message: 'No response from server. Please check your connection.',
        error: 'Network error',
        status: 0
      };
    }
    
    // Other errors - safely extract message
    let errorMessage = 'Unknown error';
    try {
      if (error && error.message) {
        errorMessage = error.message;
      } else {
        errorMessage = String(error);
      }
    } catch {
      errorMessage = 'Unknown error (could not extract message)';
    }
    
    return {
      success: false,
      message: errorMessage,
      error: errorMessage,
      status: 0
    };
  } catch (processingError) {
    // If error processing fails, return a safe error object
    try {
      console.error(`Error processing failed in handleApiError:`, safeStringify(processingError));
    } catch {
      console.error(`Error processing failed in handleApiError`);
    }
    return {
      success: false,
      message: 'Error handling failed',
      error: 'Error processing failed',
      status: 0
    };
  }
};

export const handleValidationError = (error) => {
  return {
    success: false,
    message: 'Validation failed',
    error: error.message || 'Invalid input data',
    status: 400
  };
};

export const handleNetworkError = (error) => {
  return {
    success: false,
    message: 'Network connection failed',
    error: error.message || 'Check your internet connection',
    status: 0
  };
};

export const handleError = handleApiError;

export default {
  handleApiError,
  handleValidationError,
  handleNetworkError,
  handleError
};
