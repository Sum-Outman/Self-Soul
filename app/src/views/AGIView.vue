<!--
AGI Capability View
AGI能力视图
-->
<template>
  <div class="agi-view">
    <AGICapabilityDashboard />
  </div>
</template>

<script>
import AGICapabilityDashboard from '@/components/AGICapabilityDashboard.vue';

export default {
  name: 'AGIView',
  components: {
    AGICapabilityDashboard
  }
};
</script>

<style scoped>
.agi-view {
  min-height: 100vh;
  background: #f5f7fa;
}
</style>
