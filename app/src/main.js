/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './assets/main.css'
import i18n, { detectAndSetBrowserLanguage } from './i18n.js'

// 简单的通知插件
const notifyPlugin = {
  install(app) {
    app.config.globalProperties.$notify = function(options) {
      const title = options.title || '';
      const message = options.message || '';
      const type = options.type || 'info';
      
      console.log(`[${type.toUpperCase()}] ${title}: ${message}`);
      
      // 创建一个简单的通知元素
      if (process.env.NODE_ENV !== 'production') {
        alert(`${title}: ${message}`);
      }
    };
  }
};

// 确保DOM完全加载后再挂载应用
document.addEventListener('DOMContentLoaded', () => {
    // 自动检测浏览器语言并设置
    detectAndSetBrowserLanguage()
    
    const app = createApp(App)
    app.use(router)
    app.use(i18n)
    app.use(notifyPlugin) // 添加通知插件
  
    // 确保挂载点存在
    const appContainer = document.getElementById('app');
    if (appContainer) {
      app.mount('#app')
      // 设置全局应用引用，便于错误处理器访问通知功能
      window.$app = app;
      console.log('Self Soul 应用已成功挂载！')
    } else {
      console.error('错误：未找到应用挂载点！')
    }
  })