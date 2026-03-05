/**
 * Test notification system functionality
 * This module provides a simple function to test the notification system
 */

import { notify } from '@/plugins/notification'

/**
 * Test all notification types
 */
export default function testNotifications() {
  console.log('Testing notification system...')
  
  // Test info notification
  notify.info('Information notification test', 'This is an info notification test.')
  
  // Test success notification
  notify.success('Success notification test', 'This is a success notification test.')
  
  // Test warning notification
  notify.warning('Warning notification test', 'This is a warning notification test.')
  
  // Test error notification
  notify.error('Error notification test', 'This is an error notification test.')
  
  // Test loading notification
  const loadingId = notify.loading('Loading notification test', 'This is a loading notification test.')
  
  // Close loading notification after 2 seconds
  setTimeout(() => {
    notify.close(loadingId)
    notify.success('Test completed', 'All notification types tested successfully.')
  }, 2000)
  
  console.log('Notification tests completed')
}

/**
 * Test specific notification type
 * @param {string} type - Notification type (info, success, warning, error, loading)
 */
export function testNotificationType(type) {
  const types = ['info', 'success', 'warning', 'error', 'loading']
  
  if (!types.includes(type)) {
    console.error(`Invalid notification type: ${type}. Must be one of: ${types.join(', ')}`)
    return
  }
  
  console.log(`Testing ${type} notification...`)
  
  if (type === 'loading') {
    const loadingId = notify.loading('Loading test', 'Testing loading notification...')
    setTimeout(() => {
      notify.close(loadingId)
      console.log('Loading notification test completed')
    }, 1500)
  } else {
    notify[type](`${type.charAt(0).toUpperCase() + type.slice(1)} test`, `Testing ${type} notification...`)
    console.log(`${type} notification test completed`)
  }
}