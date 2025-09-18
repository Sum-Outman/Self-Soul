/**
 * 统一的错误处理工具
 * 提供生产环境友好的错误处理机制
 */

class ErrorHandler {
  constructor() {
    this.isDevelopment = import.meta.env.MODE === 'development';
    this.errorLogs = [];
    this.maxLogs = 1000; // 最大日志数量
  }

  /**
   * 处理错误信息
   * @param {Error|string} error - 错误对象或错误消息
   * @param {string} context - 错误上下文
   * @param {Object} options - 额外选项
   */
  handleError(error, context = 'Unknown', options = {}) {
    const errorObj = error instanceof Error ? error : new Error(error);
    const errorData = {
      timestamp: new Date().toISOString(),
      context,
      message: errorObj.message,
      stack: errorObj.stack,
      ...options
    };

    // 在开发环境下显示console.error
    if (this.isDevelopment) {
      console.error(`[${context}]`, errorObj);
    }

    // 添加到错误日志
    this.addToLogs(errorData);

    // 可以根据需要发送到错误监控服务
    this.sendToMonitoringService(errorData);

    return errorData;
  }

  /**
   * 处理警告信息
   * @param {string} message - 警告消息
   * @param {string} context - 警告上下文
   */
  handleWarning(message, context = 'Unknown') {
    const warningData = {
      timestamp: new Date().toISOString(),
      context,
      message,
      type: 'warning'
    };

    // 在开发环境下显示console.warn
    if (this.isDevelopment) {
      console.warn(`[${context}]`, message);
    }

    // 添加到日志
    this.addToLogs(warningData);

    return warningData;
  }

  /**
   * 记录信息日志
   * @param {string} message - 信息消息
   * @param {string} context - 信息上下文
   */
  logInfo(message, context = 'Unknown') {
    const infoData = {
      timestamp: new Date().toISOString(),
      context,
      message,
      type: 'info'
    };

    // 在开发环境下显示console.log
    if (this.isDevelopment) {
      console.log(`[${context}]`, message);
    }

    // 添加到日志
    this.addToLogs(infoData);

    return infoData;
  }

  /**
   * 添加到日志
   * @param {Object} logData - 日志数据
   */
  addToLogs(logData) {
    this.errorLogs.push(logData);
    
    // 限制日志数量
    if (this.errorLogs.length > this.maxLogs) {
      this.errorLogs = this.errorLogs.slice(-this.maxLogs);
    }
  }

  /**
   * 发送到监控服务（可以扩展实现）
   * @param {Object} errorData - 错误数据
   */
  sendToMonitoringService(errorData) {
    // 这里可以实现发送到Sentry、LogRocket等错误监控服务
    // 示例：console.log('Sending to monitoring service:', errorData);
  }

  /**
   * 获取错误日志
   * @returns {Array} 错误日志数组
   */
  getErrorLogs() {
    return [...this.errorLogs];
  }

  /**
   * 清空错误日志
   */
  clearErrorLogs() {
    this.errorLogs = [];
  }

  /**
   * 创建错误处理函数
   * @param {string} context - 错误上下文
   * @returns {Function} 错误处理函数
   */
  createErrorHandler(context) {
    return (error, options = {}) => {
      return this.handleError(error, context, options);
    };
  }

  /**
   * 创建警告处理函数
   * @param {string} context - 警告上下文
   * @returns {Function} 警告处理函数
   */
  createWarningHandler(context) {
    return (message) => {
      return this.handleWarning(message, context);
    };
  }
}

// 创建全局实例
const errorHandler = new ErrorHandler();

// 导出默认实例和类
export default errorHandler;
export { ErrorHandler };
