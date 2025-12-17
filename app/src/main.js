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
import './assets/notificationStyles.css'
import { createPinia } from 'pinia'

// Import professional notification plugin
import notificationPlugin from './plugins/notification.js'

// 确保DOM完全加载后再挂载应用
document.addEventListener('DOMContentLoaded', () => {
    // Create app instance
    const pinia = createPinia()
    const app = createApp(App)
    
    // Register plugins
    app.use(router)
    app.use(notificationPlugin) // Add professional notification plugin
    app.use(pinia)
    
    // Global properties
    app.config.globalProperties.$isEnglish = true
    
    // 确保挂载点存在
    const appContainer = document.getElementById('app');
    if (appContainer) {
      app.mount('#app')
      console.log('Self Soul System has been successfully mounted!')
    } else {
      console.error('Error: Application mount point not found!')
    }
  })