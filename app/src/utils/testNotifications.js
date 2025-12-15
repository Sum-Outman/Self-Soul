import { notificationManager } from './notification/index.js'

/**
 * Test notification system function
 * This function tests all notification types and positions
 */
export const testNotifications = () => {
  try {
    // Clear existing notifications first
    notificationManager.closeAllNotifications()
    
    // Test info notification
    notificationManager.info({
      message: 'This is an information notification',
      position: 'top-right',
      duration: 5000
    })
    
    // Test success notification
    setTimeout(() => {
      notificationManager.success({
        message: 'This is a success notification',
        position: 'top-left',
        duration: 5000
      })
    }, 1000)
    
    // Test warning notification
    setTimeout(() => {
      notificationManager.warning({
        message: 'This is a warning notification',
        position: 'bottom-right',
        duration: 5000
      })
    }, 2000)
    
    // Test error notification
    setTimeout(() => {
      notificationManager.error({
        message: 'This is an error notification',
        position: 'bottom-left',
        duration: 5000
      })
    }, 3000)
    
    console.log('Notification system test completed')
  } catch (error) {
    console.error('Error in notification system test:', error)
    throw error
  }
}

export default testNotifications