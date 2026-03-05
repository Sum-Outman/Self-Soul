import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// Types
export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message: string
  timestamp: Date
  read: boolean
  duration?: number
  action?: {
    label: string
    callback: () => void
  }
}

export interface SidebarState {
  collapsed: boolean
  width: number
  visible: boolean
}

export interface ThemeSettings {
  mode: 'light' | 'dark' | 'auto'
  primaryColor: string
  secondaryColor: string
  accentColor: string
  fontSize: number
  fontFamily: string
  borderRadius: number
}

export interface LayoutSettings {
  sidebarPosition: 'left' | 'right'
  headerHeight: number
  footerVisible: boolean
  contentPadding: number
  gridColumns: number
}

export const useUiStore = defineStore('ui', () => {
  // State
  const notifications = ref<Notification[]>([])
  const sidebarState = ref<SidebarState>({
    collapsed: false,
    width: 240,
    visible: true
  })
  const themeSettings = ref<ThemeSettings>({
    mode: 'light',
    primaryColor: '#2196f3',
    secondaryColor: '#f50057',
    accentColor: '#00bcd4',
    fontSize: 14,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    borderRadius: 8
  })
  const layoutSettings = ref<LayoutSettings>({
    sidebarPosition: 'left',
    headerHeight: 64,
    footerVisible: true,
    contentPadding: 20,
    gridColumns: 12
  })
  const isLoading = ref(false)
  const loadingMessage = ref<string | null>(null)
  const currentView = ref<string>('home')
  const modalStack = ref<string[]>([])
  const toastQueue = ref<Notification[]>([])
  const preferences = ref<Record<string, any>>({})

  // Getters
  const unreadNotifications = computed(() => 
    notifications.value.filter(notification => !notification.read)
  )
  
  const unreadNotificationCount = computed(() => unreadNotifications.value.length)
  
  const recentNotifications = computed(() => 
    notifications.value
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, 10)
  )
  
  const sidebarWidth = computed(() => 
    sidebarState.value.collapsed ? 0 : sidebarState.value.width
  )
  
  const isSidebarVisible = computed(() => sidebarState.value.visible)
  
  const isDarkMode = computed(() => {
    if (themeSettings.value.mode === 'auto') {
      return window.matchMedia('(prefers-color-scheme: dark)').matches
    }
    return themeSettings.value.mode === 'dark'
  })
  
  const currentModal = computed(() => 
    modalStack.value.length > 0 ? modalStack.value[modalStack.value.length - 1] : null
  )
  
  const isModalOpen = computed(() => modalStack.value.length > 0)
  
  const nextToast = computed(() => 
    toastQueue.value.length > 0 ? toastQueue.value[0] : null
  )

  // Actions
  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    const newNotification: Notification = {
      ...notification,
      id: `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      read: false
    }
    
    notifications.value.unshift(newNotification)
    
    // Keep only last 100 notifications
    if (notifications.value.length > 100) {
      notifications.value = notifications.value.slice(0, 100)
    }
    
    return newNotification.id
  }

  const markNotificationAsRead = (notificationId: string) => {
    const notification = notifications.value.find(n => n.id === notificationId)
    if (notification) {
      notification.read = true
    }
  }

  const markAllNotificationsAsRead = () => {
    notifications.value.forEach(notification => {
      notification.read = true
    })
  }

  const removeNotification = (notificationId: string) => {
    notifications.value = notifications.value.filter(n => n.id !== notificationId)
  }

  const clearNotifications = () => {
    notifications.value = []
  }

  const toggleSidebar = () => {
    sidebarState.value.collapsed = !sidebarState.value.collapsed
  }

  const setSidebarVisible = (visible: boolean) => {
    sidebarState.value.visible = visible
  }

  const setSidebarWidth = (width: number) => {
    sidebarState.value.width = Math.max(200, Math.min(400, width))
  }

  const updateTheme = (themeUpdates: Partial<ThemeSettings>) => {
    themeSettings.value = { ...themeSettings.value, ...themeUpdates }
    
    // Apply theme changes to document
    if (themeUpdates.mode !== undefined || themeUpdates.primaryColor !== undefined) {
      applyThemeToDocument()
    }
  }

  const updateLayout = (layoutUpdates: Partial<LayoutSettings>) => {
    layoutSettings.value = { ...layoutSettings.value, ...layoutUpdates }
  }

  const setLoading = (loading: boolean, message?: string) => {
    isLoading.value = loading
    loadingMessage.value = message || null
  }

  const setCurrentView = (view: string) => {
    currentView.value = view
  }

  const openModal = (modalId: string) => {
    if (!modalStack.value.includes(modalId)) {
      modalStack.value.push(modalId)
    }
  }

  const closeModal = (modalId: string) => {
    modalStack.value = modalStack.value.filter(id => id !== modalId)
  }

  const closeAllModals = () => {
    modalStack.value = []
  }

  const showToast = (toast: Omit<Notification, 'id' | 'timestamp' | 'read'>) => {
    const newToast: Notification = {
      ...toast,
      id: `toast_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
      read: false
    }
    
    toastQueue.value.push(newToast)
    return newToast.id
  }

  const removeToast = (toastId: string) => {
    toastQueue.value = toastQueue.value.filter(toast => toast.id !== toastId)
  }

  const clearToastQueue = () => {
    toastQueue.value = []
  }

  const setPreference = (key: string, value: any) => {
    preferences.value[key] = value
    savePreferences()
  }

  const getPreference = <T>(key: string, defaultValue: T): T => {
    return preferences.value[key] !== undefined ? preferences.value[key] : defaultValue
  }

  const resetPreferences = () => {
    preferences.value = {}
    localStorage.removeItem('ui_preferences')
  }

  // Helper functions
  const applyThemeToDocument = () => {
    const root = document.documentElement
    
    // Apply color scheme
    if (isDarkMode.value) {
      root.style.setProperty('--color-scheme', 'dark')
      root.classList.add('dark-mode')
      root.classList.remove('light-mode')
    } else {
      root.style.setProperty('--color-scheme', 'light')
      root.classList.add('light-mode')
      root.classList.remove('dark-mode')
    }
    
    // Apply theme colors
    root.style.setProperty('--primary-color', themeSettings.value.primaryColor)
    root.style.setProperty('--secondary-color', themeSettings.value.secondaryColor)
    root.style.setProperty('--accent-color', themeSettings.value.accentColor)
    root.style.setProperty('--font-size', `${themeSettings.value.fontSize}px`)
    root.style.setProperty('--font-family', themeSettings.value.fontFamily)
    root.style.setProperty('--border-radius', `${themeSettings.value.borderRadius}px`)
  }

  const savePreferences = () => {
    try {
      localStorage.setItem('ui_preferences', JSON.stringify({
        themeSettings: themeSettings.value,
        layoutSettings: layoutSettings.value,
        preferences: preferences.value
      }))
    } catch (error) {
      console.error('Failed to save UI preferences:', error)
    }
  }

  const loadPreferences = () => {
    try {
      const saved = localStorage.getItem('ui_preferences')
      if (saved) {
        const parsed = JSON.parse(saved)
        if (parsed.themeSettings) themeSettings.value = parsed.themeSettings
        if (parsed.layoutSettings) layoutSettings.value = parsed.layoutSettings
        if (parsed.preferences) preferences.value = parsed.preferences
        
        // Apply loaded theme
        applyThemeToDocument()
      }
    } catch (error) {
      console.error('Failed to load UI preferences:', error)
    }
  }

  const resetUiState = () => {
    notifications.value = []
    sidebarState.value = {
      collapsed: false,
      width: 240,
      visible: true
    }
    themeSettings.value = {
      mode: 'light',
      primaryColor: '#2196f3',
      secondaryColor: '#f50057',
      accentColor: '#00bcd4',
      fontSize: 14,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
      borderRadius: 8
    }
    layoutSettings.value = {
      sidebarPosition: 'left',
      headerHeight: 64,
      footerVisible: true,
      contentPadding: 20,
      gridColumns: 12
    }
    isLoading.value = false
    loadingMessage.value = null
    currentView.value = 'home'
    modalStack.value = []
    toastQueue.value = []
    preferences.value = {}
  }

  // Initialize
  loadPreferences()

  return {
    // State
    notifications,
    sidebarState,
    themeSettings,
    layoutSettings,
    isLoading,
    loadingMessage,
    currentView,
    modalStack,
    toastQueue,
    preferences,
    
    // Getters
    unreadNotifications,
    unreadNotificationCount,
    recentNotifications,
    sidebarWidth,
    isSidebarVisible,
    isDarkMode,
    currentModal,
    isModalOpen,
    nextToast,
    
    // Actions
    addNotification,
    markNotificationAsRead,
    markAllNotificationsAsRead,
    removeNotification,
    clearNotifications,
    toggleSidebar,
    setSidebarVisible,
    setSidebarWidth,
    updateTheme,
    updateLayout,
    setLoading,
    setCurrentView,
    openModal,
    closeModal,
    closeAllModals,
    showToast,
    removeToast,
    clearToastQueue,
    setPreference,
    getPreference,
    resetPreferences,
    resetUiState,
    
    // Helper functions
    applyThemeToDocument,
    savePreferences,
    loadPreferences
  }
})