import axios from 'axios';
import errorHandler from './errorHandler';

// 创建axios实例
const api = axios.create({
  // 根据系统要求，主API网关端口为8000
  baseURL: 'http://localhost:8000',
  timeout: 10000, // 10秒超时
  headers: {
    'Content-Type': 'application/json',
  }
});

// 请求拦截器
api.interceptors.request.use(
  config => {
    // 可以在这里添加认证token等
    return config;
  },
  error => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  response => {
    return response;
  },
  error => {
    // 统一处理错误
    if (error.code === 'ECONNREFUSED') {
      errorHandler.handleError(
        error,
        'Backend service not running. Please start the Python backend with "python core/main.py".'
      );
    } else {
      errorHandler.handleError(error, 'API request failed');
    }
    return Promise.reject(error);
  }
);

export default api;