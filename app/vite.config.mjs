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
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    vue(),
    // 构建分析插件，仅在分析模式或生产构建时启用
    visualizer({
      filename: '../dist/stats.html',
      open: process.env.ANALYZE === 'true',
      gzipSize: true,
      brotliSize: true,
      template: 'treemap', // sunburst, treemap, network, raw-data
    })
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
    // Ensure JSON files can be parsed correctly
    extensions: ['.js', '.json', '.vue', '.ts']
  },
  base: './', // Ensure resources use relative paths
  build: {
    outDir: '../dist', // Explicit output directory
    assetsDir: 'assets', // Specify assets directory
    sourcemap: true, // Generate sourcemap for debugging
    minify: 'terser', // Enable minification with terser
    terserOptions: {
      compress: {
        drop_console: true, // Remove console logs in production
        drop_debugger: true, // Remove debugger statements
        pure_funcs: ['console.log', 'console.info', 'console.debug', 'console.trace'], // Remove specific console functions
        dead_code: true, // Remove dead code
        unused: true, // Drop unreferenced functions and variables
        reduce_vars: true, // Optimize variable usage
        booleans: true, // Optimize boolean expressions
        conditionals: true, // Optimize conditionals
        evaluate: true, // Evaluate constant expressions
        if_return: true, // Optimize if-return sequences
        join_vars: true, // Join consecutive variable declarations
        collapse_vars: true, // Collapse single-use variables
        sequences: true, // Use comma operator where possible
        side_effects: true // Remove pure function calls with no side effects
      },
      mangle: {
        safari10: true // Work around Safari 10/11 bugs in function scoping
      },
      format: {
        comments: false // Remove all comments
      }
    },
    rollupOptions: {
      output: {
        entryFileNames: 'assets/[name].[hash].js',
        chunkFileNames: 'assets/[name].[hash].js',
        assetFileNames: 'assets/[name].[hash].[ext]',
        manualChunks(id) {
          // Vendor chunk splitting for better caching
          if (id.includes('node_modules')) {
            if (id.includes('vue')) {
              return 'vue-vendor'
            }
            if (id.includes('axios')) {
              return 'axios-vendor'
            }
            if (id.includes('chart.js')) {
              return 'chartjs-vendor'
            }
            if (id.includes('@vueuse')) {
              return 'vueuse-vendor'
            }
            if (id.includes('lodash') || id.includes('lodash-es')) {
              return 'lodash-vendor'
            }
            // 将其他大型库分组
            if (id.includes('date-fns') || id.includes('moment')) {
              return 'date-vendor'
            }
            return 'vendor' // Other third-party libraries
          }
          
          // 按视图分割业务代码以提高缓存效率
          if (id.includes('/views/')) {
            const match = id.match(/\/views\/([^\/]+)\.vue$/)
            if (match) {
              const viewName = match[1]
              // 将相关视图分组
              if (['HomeView', 'Conversation'].includes(viewName)) {
                return 'chat-views'
              }
              if (['SettingsView', 'RobotSettingsView'].includes(viewName)) {
                return 'settings-views'
              }
              if (['KnowledgeView', 'TrainView'].includes(viewName)) {
                return 'knowledge-views'
              }
              return `view-${viewName.toLowerCase()}`
            }
          }
        }
      }
    },
    chunkSizeWarningLimit: 1000, // Set chunk size warning limit to 1000KB
    reportCompressedSize: false // Disable compressed size reporting for faster builds
  },
  server: {
    host: '0.0.0.0', // Allow external access
    port: 5175, // Restore default port
    open: true, // Open browser automatically after startup
    strictPort: true, // Allow automatic port switching
    hmr: {
      overlay: true // Display error overlay
    },
    // Configure proxy to ensure frontend can access backend API
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path,
        secure: false
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path,
        secure: false
      },
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path,
        secure: false
      }
    }
  }
})
