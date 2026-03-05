<template>
  <div class="hardware-category">
    <h5>{{ title }} ({{ items.length }})</h5>
    <div class="hardware-items">
      <div v-for="item in items" :key="item.id" class="hardware-item" :class="getItemStatus(item)">
        <span v-for="field in fieldConfig.fields" :key="field.key" :class="field.cssClass">
          {{ field.format ? field.format(item[field.key], item) : item[field.key] }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  items: {
    type: Array,
    required: true
  },
  itemType: {
    type: String,
    required: true,
    validator: (value) => ['sensor-instance', 'camera-instance', 'device-instance', 'connection', 
                         'sensor-type', 'camera-type', 'device-type', 'connection-type', 'input-type'].includes(value)
  },
  status: {
    type: String,
    default: 'connected',
    validator: (value) => ['connected', 'available', 'disconnected'].includes(value)
  },
  statusGetter: {
    type: Function,
    default: null
  }
})

// Determine which fields to display based on itemType
const getFieldConfig = () => {
  const configs = {
    'sensor-instance': {
      fields: [
        { key: 'id', label: 'ID', cssClass: 'item-id' },
        { key: 'description', label: 'Description', cssClass: 'item-description' },
        { key: 'config', label: 'Config', cssClass: 'item-config', format: (value) => `Config: ${value}` }
      ]
    },
    'camera-instance': {
      fields: [
        { key: 'id', label: 'ID', cssClass: 'item-id' },
        { key: 'description', label: 'Description', cssClass: 'item-description' },
        { key: 'resolution', label: 'Resolution', cssClass: 'item-config', format: (value, item) => `Resolution: ${item.resolution}, ${item.fps} FPS` }
      ]
    },
    'device-instance': {
      fields: [
        { key: 'id', label: 'ID', cssClass: 'item-id' },
        { key: 'description', label: 'Description', cssClass: 'item-description' },
        { key: 'nature', label: 'Nature', cssClass: 'item-config', format: (value, item) => `Nature: ${item.nature}, Protocol: ${item.protocol}` }
      ]
    },
    'connection': {
      fields: [
        { key: 'id', label: 'ID', cssClass: 'item-id' },
        { key: 'description', label: 'Description', cssClass: 'item-description' },
        { key: 'type', label: 'Type', cssClass: 'item-config', format: (value, item) => `${item.type}:${item.port}, ${item.enabled ? 'Enabled' : 'Disabled'}` }
      ]
    },
    'sensor-type': {
      fields: [
        { key: 'name', label: 'Name', cssClass: 'item-name' },
        { key: 'type', label: 'Type', cssClass: 'item-type' },
        { key: 'count', label: 'Quantity', cssClass: 'item-count', format: (value, item) => `Quantity: ${item.count}/${item.maxCount}` },
        { key: 'description', label: 'Description', cssClass: 'item-description' }
      ]
    },
    'camera-type': {
      fields: [
        { key: 'name', label: 'Name', cssClass: 'item-name' },
        { key: 'type', label: 'Type', cssClass: 'item-type' },
        { key: 'count', label: 'Quantity', cssClass: 'item-count', format: (value, item) => `Quantity: ${item.count}/${item.maxCount}` },
        { key: 'description', label: 'Description', cssClass: 'item-description' }
      ]
    },
    'device-type': {
      fields: [
        { key: 'name', label: 'Name', cssClass: 'item-name' },
        { key: 'count', label: 'Quantity', cssClass: 'item-count', format: (value, item) => `Quantity: ${item.count}/${item.maxCount}` },
        { key: 'description', label: 'Description', cssClass: 'item-description' }
      ]
    },
    'connection-type': {
      fields: [
        { key: 'name', label: 'Name', cssClass: 'item-name' },
        { key: 'count', label: 'Max Ports', cssClass: 'item-count', format: (value, item) => `Max Ports: ${item.maxPorts}` },
        { key: 'description', label: 'Description', cssClass: 'item-description' }
      ]
    },
    'input-type': {
      fields: [
        { key: 'name', label: 'Name', cssClass: 'item-name' },
        { key: 'description', label: 'Description', cssClass: 'item-description' }
      ]
    }
  }
  
  return configs[props.itemType] || configs['sensor-instance']
}

const fieldConfig = computed(() => getFieldConfig())

// Function to get status for an item
const getItemStatus = (item) => {
  if (props.statusGetter && typeof props.statusGetter === 'function') {
    return props.statusGetter(item)
  }
  return props.status
}
</script>

<style scoped>
.hardware-category {
  margin-bottom: 20px;
}

.hardware-category h5 {
  margin-bottom: 10px;
  font-size: 1.1em;
  color: #333;
  border-bottom: 1px solid #e0e0e0;
  padding-bottom: 5px;
}

.hardware-items {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.hardware-item {
  padding: 10px;
  border: 1px solid #e0e0e0;
  border-radius: 4px;
  background-color: #f9f9f9;
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  align-items: center;
}

.hardware-item.connected {
  border-left: 4px solid #28a745;
}

.hardware-item.available {
  border-left: 4px solid #17a2b8;
}

.hardware-item.disconnected {
  border-left: 4px solid #dc3545;
}

.item-id {
  font-family: monospace;
  font-size: 0.9em;
  color: #666;
  background-color: #f0f0f0;
  padding: 2px 6px;
  border-radius: 3px;
}

.item-description {
  font-weight: 500;
  color: #333;
}

.item-config {
  font-size: 0.9em;
  color: #6c757d;
}

.item-name {
  font-weight: 500;
  color: #333;
}

.item-type {
  font-size: 0.9em;
  color: #6c757d;
  font-style: italic;
}

.item-count {
  font-size: 0.9em;
  color: #17a2b8;
  font-weight: 500;
}
</style>