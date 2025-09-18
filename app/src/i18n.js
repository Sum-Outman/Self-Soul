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

import { createI18n } from 'vue-i18n'

// 语言文件占位符，将在运行时动态加载
const messages = {
  zh: {},
  en: {},
  de: {},
  ja: {},
  ru: {}
}

// 统一的多语言词典管理系统
const i18n = createI18n({
  legacy: false,  // 修改为false，支持composition API
  locale: localStorage.getItem('user-language') || 'zh', // 默认使用中文
  fallbackLocale: 'en',
  messages
})

// 同步加载语言文件（确保在应用启动前完成）
export const loadLanguageFiles = () => {
  try {
    // 直接导入所有语言文件
    import('./locales/zh.json').then(module => {
      i18n.global.setLocaleMessage('zh', module.default)
    })
    import('./locales/en.json').then(module => {
      i18n.global.setLocaleMessage('en', module.default)
    })
    import('./locales/de.json').then(module => {
      i18n.global.setLocaleMessage('de', module.default)
    })
    import('./locales/ja.json').then(module => {
      i18n.global.setLocaleMessage('ja', module.default)
    })
    import('./locales/ru.json').then(module => {
      i18n.global.setLocaleMessage('ru', module.default)
    })
  } catch (error) {
    console.error('加载语言文件时出错:', error)
  }
}

// 立即加载语言文件
loadLanguageFiles()

// 添加语言切换函数
export const switchLanguage = (lang) => {
  if (['zh', 'en', 'de', 'ja', 'ru'].includes(lang)) {
    i18n.global.locale.value = lang
    localStorage.setItem('user-language', lang)
    // 通知所有组件语言已更改
    window.dispatchEvent(new CustomEvent('language-changed', { detail: { lang } }))
    return true
  }
  return false
}

// 自动检测浏览器语言并设置语言（现在作为函数导出，由应用初始化时调用）
export const detectAndSetBrowserLanguage = () => {
  try {
    const savedLang = localStorage.getItem('user-language')
    // 如果已经保存了用户语言偏好，则不覆盖
    if (!savedLang) {
      const browserLang = navigator.language.split('-')[0]
      if (['zh', 'en', 'de', 'ja', 'ru'].includes(browserLang)) {
        i18n.global.locale.value = browserLang
        localStorage.setItem('user-language', browserLang)
      }
    }
  } catch (error) {
    console.warn('检测浏览器语言时出错:', error)
  }
}

// 根据输入文本检测语言
export const detectInputLanguage = (text) => {
  if (!text) return i18n.global.locale.value;
  
  // 中文检测
  if (/[\u4e00-\u9fa5]/.test(text)) return 'zh';
  
  // 日文检测
  if (/[\u3040-\u309F\u30A0-\u30FF]/.test(text)) return 'ja';
  
  // 俄文检测
  if (/[\u0400-\u04FF]/.test(text)) return 'ru';
  
  // 德文检测
  if (/[äöüßÄÖÜ]/.test(text)) return 'de';
  
  return 'en';
}

// 将辅助函数添加到i18n实例
i18n.switchLanguage = switchLanguage
i18n.detectInputLanguage = detectInputLanguage
i18n.detectAndSetBrowserLanguage = detectAndSetBrowserLanguage

export default i18n
