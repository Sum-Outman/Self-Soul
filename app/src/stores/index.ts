import { defineStore } from 'pinia'

// Export all stores from this file
export * from './training'
export * from './system'
export * from './models'
export * from './robot'
export * from './ui'

// Helper function to create a store with default options
export const createStore = (id: string, options: any) => {
  return defineStore(id, options)
}