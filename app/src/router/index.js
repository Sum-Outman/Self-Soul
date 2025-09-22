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

import { createRouter, createWebHashHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/HomeView.vue'),
    alias: ['/index.html', '/home', '/main']
  },
  {
    path: '/training',
    name: 'Training',
    component: () => import('../views/TrainView.vue')
  },
  {
    path: '/knowledge',
    name: 'Knowledge',
    component: () => import('../views/KnowledgeView.vue')
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('../views/SettingsView.vue')
  },
  {
    path: '/help',
    name: 'Help',    
    component: () => import('../views/HelpView.vue')
  },
  {
    path: '/notification-test',
    name: 'NotificationTest',
    component: () => import('../components/NotificationTester.vue')
  },
  // 确保所有未知路径重定向到首页
  {
    path: '/:pathMatch(.*)*',
    redirect: '/' 
  }
]

const router = createRouter({
  // 使用哈希路由确保兼容性
  history: createWebHashHistory(),
  routes
})

// 添加全局前置守卫确保正确加载
router.beforeEach((to, from, next) => {
  // 只在开发环境下输出日志
  if (import.meta.env.DEV) {
    console.log(`Navigating to: ${to.path}`)
  }
  next()
})

export default router
