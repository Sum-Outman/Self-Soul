import { createRouter, createWebHashHistory } from 'vue-router'


const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/HomeView.vue'),
    alias: ['/index.html', '/home', '/main']
  },
  {
    path: '/conversation',
    name: 'Conversation',
    component: () => import('../views/Conversation.vue')
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

  // Ensure all unknown paths redirect to home
  {
    path: '/:pathMatch(.*)*',
    redirect: '/' 
  }
]

const router = createRouter({
  // Use hash router for compatibility
  history: createWebHashHistory(),
  routes
})

// Add global before guard to ensure correct loading
router.beforeEach((to, from, next) => {
  // Only log in development environment
  if (import.meta.env.DEV) {
    console.log(`Navigating to: ${to.path}`)
  }
  next()
})

export default router
