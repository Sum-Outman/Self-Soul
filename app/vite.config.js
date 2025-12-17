/*!
 * AGI Soul System
 * Copyright 2025 AGI Soul Team
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

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
    // 确保JSON文件可以被正确解析
    extensions: ['.js', '.json', '.vue', '.ts']
  },
  base: './', // 确保资源使用相对路径
  build: {
    outDir: '../dist', // 明确输出目录
    assetsDir: 'assets', // 指定资源目录
    sourcemap: true, // 生成sourcemap便于调试
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name].js',
        chunkFileNames: 'assets/[name].js',
        assetFileNames: 'assets/[name].[ext]'
      }
    }
  },
  server: {
    host: '0.0.0.0', // 允许外部访问
    port: 5175, // 使用备用端口
    open: true, // 启动后自动打开浏览器
    strictPort: false, // 允许自动切换端口
    hmr: {
      overlay: true // 显示错误覆盖层
    },
    // 设置代理配置，确保前端可以访问后端API
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path
      },
      '/health': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ws/, '')
      }
    }
  }
})
