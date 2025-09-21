import { createApp } from 'vue'
import Notification from '../components/Notification.vue'

// 通知管理类
class NotificationManager {
  constructor() {
    this.notifications = []
    this.maxVisible = 5
    this.container = null
  }

  // 初始化通知容器
  init() {
    if (!this.container) {
      this.container = document.createElement('div')
      this.container.className = 'notification-container'
      document.body.appendChild(this.container)
    }
  }

  // 创建通知
  create(options = {}) {
    const { 
      title = '',
      message = '',
      type = 'info',
      duration = 3000,
      closable = true,
      position = 'top-right'
    } = options

    if (!message) return

    // 初始化容器
    this.init()

    // 移除超出最大数量的旧通知
    if (this.notifications.length >= this.maxVisible) {
      this.removeNotification(this.notifications[0])
    }

    // 创建通知实例
    const notificationOptions = {
      title,
      message,
      type,
      duration,
      closable,
      position,
      onClose: () => {
        this.removeNotification(notificationInstance)
      }
    }

    const notificationApp = createApp(Notification, notificationOptions)
    const notificationWrapper = document.createElement('div')
    this.container.appendChild(notificationWrapper)
    
    notificationApp.mount(notificationWrapper)

    const notificationInstance = {
      app: notificationApp,
      wrapper: notificationWrapper,
      options: notificationOptions
    }

    this.notifications.push(notificationInstance)

    return {
      close: () => {
        this.removeNotification(notificationInstance)
      }
    }
  }

  // 移除通知
  removeNotification(instance) {
    const index = this.notifications.indexOf(instance)
    if (index !== -1) {
      this.notifications.splice(index, 1)
      
      if (instance.wrapper.parentNode) {
        instance.wrapper.parentNode.removeChild(instance.wrapper)
      }
      
      if (instance.app && instance.app.unmount) {
        instance.app.unmount()
      }
    }
  }

  // 关闭所有通知
  closeAll() {
    this.notifications.forEach(instance => {
      this.removeNotification(instance)
    })
  }
}

// 创建全局通知管理器实例
const notificationManager = new NotificationManager()

// 创建通知插件
const notificationPlugin = {
  install(app) {
    // 添加全局方法
    app.config.globalProperties.$notify = function(options) {
      return notificationManager.create(options)
    }
    
    // 添加到app实例属性
    app.notificationManager = notificationManager
  }
}

// 导出插件和管理器

export default notificationPlugin
export { notificationManager }

// 直接导出通知方法
export const notify = {
  info(options) {
    return notificationManager.create(typeof options === 'string' ? { message: options, type: 'info' } : { ...options, type: 'info' })
  },
  success(options) {
    return notificationManager.create(typeof options === 'string' ? { message: options, type: 'success' } : { ...options, type: 'success' })
  },
  warning(options) {
    return notificationManager.create(typeof options === 'string' ? { message: options, type: 'warning' } : { ...options, type: 'warning' })
  },
  error(options) {
    return notificationManager.create(typeof options === 'string' ? { message: options, type: 'error' } : { ...options, type: 'error' })
  }
}