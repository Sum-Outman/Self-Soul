// Simple notification test script

// This script tests the notification system by displaying different types of notifications
// with various configurations. It's designed to be imported and used in a component
// to verify that the notification system works correctly.

import { notify } from '../plugins/notification.js'

// Test function to show all notification types
function testNotifications() {
  console.log('Running notification system tests...')
  
  // Test info notification
  notify.info({
    title: 'Information',
    message: 'This is a test info notification',
    duration: 3000,
    position: 'top-right'
  })
  
  // Test success notification
  notify.success({
    title: 'Success',
    message: 'This is a test success notification',
    duration: 4000,
    position: 'top-left'
  })
  
  // Test warning notification
  notify.warning({
    title: 'Warning',
    message: 'This is a test warning notification',
    duration: 5000,
    position: 'bottom-right'
  })
  
  // Test error notification
  notify.error({
    title: 'Error',
    message: 'This is a test error notification',
    duration: 6000,
    position: 'bottom-left'
  })
  
  // Test notification without title
  setTimeout(() => {
    notify.info({
      message: 'This notification has no title',
      duration: 3000,
      position: 'top-center'
    })
  }, 1000)
  
  // Test notification with custom duration
  setTimeout(() => {
    notify.success({
      title: 'Long Duration',
      message: 'This notification stays for 10 seconds',
      duration: 10000,
      position: 'bottom-center'
    })
  }, 2000)
  
  // Test notification with minimal configuration (just message)
  setTimeout(() => {
    notify.info('Minimal notification with just a message')
  }, 3000)
  
  console.log('Notification tests completed. Check UI for results.')
}

// Export the test function
export default testNotifications