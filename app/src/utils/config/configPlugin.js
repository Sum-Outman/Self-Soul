/**
 * 配置管理器Vue插件
 * 提供全局配置访问能力
 */

import configManager from './configManager';

const ConfigPlugin = {
  install(app, options = {}) {
    // 添加全局属性
    app.config.globalProperties.$config = configManager;
    
    // 添加全局方法
    app.config.globalProperties.$getConfig = (path, defaultValue) => {
      return configManager.get(path, defaultValue);
    };
    
    // 添加全局方法：获取前端URL
    app.config.globalProperties.$getFrontendUrl = () => {
      return configManager.getFrontendUrl();
    };
    
    // 添加全局方法：获取后端文档URL
    app.config.globalProperties.$getBackendDocsUrl = () => {
      return configManager.getBackendDocsUrl();
    };
    
    // 添加全局方法：获取后端API URL
    app.config.globalProperties.$getBackendUrl = (endpoint = '') => {
      return configManager.getBackendUrl(endpoint);
    };
    
    // 提供/注入配置管理器
    app.provide('config', configManager);
    
    // 注册全局组件（如果需要）
    // app.component('ConfigDisplay', ...);
  }
};

// 导出插件和配置管理器
export { ConfigPlugin, configManager };
export default ConfigPlugin;