<!--
AGI Capability Dashboard Component
AGI能力仪表板组件
-->
<template>
  <div class="agi-dashboard">
    <div class="dashboard-header">
      <h2>AGI Capability Dashboard</h2>
      <div class="header-actions">
        <button @click="refreshData" class="refresh-btn" :disabled="loading">
          {{ loading ? 'Loading...' : 'Refresh' }}
        </button>
      </div>
    </div>

    <div class="dashboard-grid">
      <div class="card status-card">
        <h3>Framework Status</h3>
        <div class="status-grid">
          <div class="status-item" :class="{ active: frameworkStatus.enabled }">
            <span class="status-icon">{{ frameworkStatus.enabled ? '✓' : '✗' }}</span>
            <span class="status-label">AGI Enhancement</span>
          </div>
          <div class="status-item" :class="{ active: frameworkStatus.frameworks?.pdac_loop?.available }">
            <span class="status-icon">{{ frameworkStatus.frameworks?.pdac_loop?.available ? '✓' : '✗' }}</span>
            <span class="status-label">PDAC Loop</span>
          </div>
          <div class="status-item" :class="{ active: frameworkStatus.frameworks?.performance_evaluation?.available }">
            <span class="status-icon">{{ frameworkStatus.frameworks?.performance_evaluation?.available ? '✓' : '✗' }}</span>
            <span class="status-label">Performance Eval</span>
          </div>
          <div class="status-item" :class="{ active: frameworkStatus.frameworks?.self_learning_evolution?.available }">
            <span class="status-icon">{{ frameworkStatus.frameworks?.self_learning_evolution?.available ? '✓' : '✗' }}</span>
            <span class="status-label">Self-Learning</span>
          </div>
        </div>
      </div>

      <div class="card maturity-card">
        <h3>AGI Maturity Level</h3>
        <div class="maturity-display">
          <div class="maturity-score">
            <span class="score-value">{{ maturityScore.toFixed(2) }}</span>
            <span class="score-max">/ 1.0</span>
          </div>
          <div class="maturity-level">{{ maturityLevel }}</div>
          <div class="maturity-bar">
            <div class="maturity-progress" :style="{ width: (maturityScore * 100) + '%' }"></div>
          </div>
        </div>
      </div>

      <div class="card capabilities-card">
        <h3>Core Capabilities</h3>
        <div class="capabilities-list">
          <div v-for="(cap, name) in capabilities" :key="name" class="capability-item">
            <div class="capability-header">
              <span class="capability-name">{{ formatCapabilityName(name) }}</span>
              <span class="capability-weight">({{ (cap.weight * 100).toFixed(0) }}%)</span>
            </div>
            <div class="capability-bar">
              <div class="capability-progress" :style="{ width: (cap.score * 100) + '%' }"></div>
            </div>
            <span class="capability-score">{{ (cap.score * 100).toFixed(0) }}%</span>
          </div>
        </div>
      </div>

      <div class="card milestones-card">
        <h3>Milestones</h3>
        <div class="milestones-list">
          <div v-for="milestone in milestones" :key="milestone.name" 
               class="milestone-item" :class="{ achieved: milestone.achieved }">
            <span class="milestone-icon">{{ milestone.achieved ? '★' : '○' }}</span>
            <div class="milestone-info">
              <span class="milestone-name">{{ formatMilestoneName(milestone.name) }}</span>
              <span class="milestone-threshold">Threshold: {{ (milestone.threshold * 100).toFixed(0) }}%</span>
            </div>
          </div>
        </div>
      </div>

      <div class="card actions-card">
        <h3>Actions</h3>
        <div class="actions-list">
          <button @click="runSelfImprovement" class="action-btn" :disabled="improving">
            {{ improving ? 'Running...' : 'Run Self-Improvement' }}
          </button>
          <button @click="processInput" class="action-btn">
            Test PDAC Loop
          </button>
          <button @click="exportReport" class="action-btn secondary">
            Export Report
          </button>
        </div>
        <div v-if="lastAction" class="action-result">
          <h4>Last Action Result</h4>
          <pre>{{ lastAction }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import api from '@/utils/api';

export default {
  name: 'AGICapabilityDashboard',
  data() {
    return {
      loading: false,
      improving: false,
      frameworkStatus: {},
      maturityScore: 0,
      maturityLevel: 'PROTOTYPE',
      capabilities: {},
      milestones: [],
      lastAction: null
    };
  },
  async mounted() {
    await this.refreshData();
  },
  methods: {
    async refreshData() {
      this.loading = true;
      try {
        await Promise.all([
          this.fetchFrameworkStatus(),
          this.fetchCapabilities(),
          this.fetchMilestones()
        ]);
      } catch (error) {
        console.error('Failed to refresh data:', error);
      } finally {
        this.loading = false;
      }
    },
    async fetchFrameworkStatus() {
      try {
        const response = await api.get('/api/agi/enhancement/status');
        if (response.data.status === 'success') {
          this.frameworkStatus = response.data.data;
        }
      } catch (error) {
        console.error('Failed to fetch framework status:', error);
      }
    },
    async fetchCapabilities() {
      try {
        const [capsResponse, assessmentResponse] = await Promise.all([
          api.get('/api/agi/capabilities/list'),
          api.get('/api/agi/capabilities/assessment')
        ]);
        
        if (capsResponse.data.status === 'success') {
          this.capabilities = capsResponse.data.data.capabilities;
        }
        
        if (assessmentResponse.data.status === 'success') {
          const assessment = assessmentResponse.data.data;
          this.maturityScore = assessment.system_evaluation?.overall_agi_score || 0;
          this.maturityLevel = assessment.system_evaluation?.maturity_level || 'PROTOTYPE';
          
          const avgScores = assessment.system_evaluation?.capability_averages || {};
          for (const [name, score] of Object.entries(avgScores)) {
            if (this.capabilities[name]) {
              this.capabilities[name].score = score;
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch capabilities:', error);
      }
    },
    async fetchMilestones() {
      try {
        const response = await api.get('/api/agi/milestones');
        if (response.data.status === 'success') {
          const data = response.data.data;
          this.milestones = [...data.achieved, ...data.pending];
        }
      } catch (error) {
        console.error('Failed to fetch milestones:', error);
      }
    },
    async runSelfImprovement() {
      this.improving = true;
      try {
        const response = await api.post('/api/agi/self-improvement/run', {});
        this.lastAction = JSON.stringify(response.data, null, 2);
        await this.refreshData();
      } catch (error) {
        this.lastAction = `Error: ${error.message}`;
      } finally {
        this.improving = false;
      }
    },
    async processInput() {
      try {
        const response = await api.post('/api/agi/enhancement/process', {
          input: 'Test input for PDAC loop',
          perception_type: 'textual'
        });
        this.lastAction = JSON.stringify(response.data, null, 2);
      } catch (error) {
        this.lastAction = `Error: ${error.message}`;
      }
    },
    async exportReport() {
      try {
        const response = await api.get('/api/agi/maturity/progress');
        const report = JSON.stringify(response.data, null, 2);
        const blob = new Blob([report], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `agi-report-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } catch (error) {
        this.lastAction = `Error: ${error.message}`;
      }
    },
    formatCapabilityName(name) {
      return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    },
    formatMilestoneName(name) {
      return name.split('_').map(word => 
        word.charAt(0).toUpperCase() + word.slice(1)
      ).join(' ');
    }
  }
};
</script>

<style scoped>
.agi-dashboard {
  padding: 20px;
  max-width: 1400px;
  margin: 0 auto;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.dashboard-header h2 {
  margin: 0;
  color: #333;
}

.refresh-btn {
  padding: 8px 16px;
  background: #4CAF50;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.refresh-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
}

.card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.card h3 {
  margin: 0 0 15px 0;
  color: #333;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px;
  background: #f5f5f5;
  border-radius: 4px;
}

.status-item.active {
  background: #e8f5e9;
}

.status-icon {
  font-size: 18px;
  color: #f44336;
}

.status-item.active .status-icon {
  color: #4CAF50;
}

.status-label {
  font-size: 14px;
}

.maturity-display {
  text-align: center;
}

.maturity-score {
  font-size: 48px;
  font-weight: bold;
  color: #2196F3;
}

.score-max {
  font-size: 24px;
  color: #999;
}

.maturity-level {
  font-size: 18px;
  color: #666;
  margin: 10px 0;
}

.maturity-bar {
  height: 10px;
  background: #e0e0e0;
  border-radius: 5px;
  overflow: hidden;
}

.maturity-progress {
  height: 100%;
  background: linear-gradient(90deg, #4CAF50, #2196F3);
  transition: width 0.5s ease;
}

.capabilities-list {
  max-height: 300px;
  overflow-y: auto;
}

.capability-item {
  margin-bottom: 12px;
}

.capability-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.capability-name {
  font-size: 14px;
  font-weight: 500;
}

.capability-weight {
  font-size: 12px;
  color: #999;
}

.capability-bar {
  height: 6px;
  background: #e0e0e0;
  border-radius: 3px;
  overflow: hidden;
}

.capability-progress {
  height: 100%;
  background: #2196F3;
  transition: width 0.3s ease;
}

.capability-score {
  font-size: 12px;
  color: #666;
}

.milestones-list {
  max-height: 250px;
  overflow-y: auto;
}

.milestone-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px;
  border-radius: 4px;
  margin-bottom: 8px;
}

.milestone-item.achieved {
  background: #e8f5e9;
}

.milestone-icon {
  font-size: 20px;
  color: #ccc;
}

.milestone-item.achieved .milestone-icon {
  color: #FFD700;
}

.milestone-info {
  display: flex;
  flex-direction: column;
}

.milestone-name {
  font-size: 14px;
  font-weight: 500;
}

.milestone-threshold {
  font-size: 12px;
  color: #999;
}

.actions-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.action-btn {
  padding: 12px;
  background: #2196F3;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
  transition: background 0.3s;
}

.action-btn:hover {
  background: #1976D2;
}

.action-btn:disabled {
  background: #ccc;
  cursor: not-allowed;
}

.action-btn.secondary {
  background: #607D8B;
}

.action-btn.secondary:hover {
  background: #455A64;
}

.action-result {
  margin-top: 15px;
  padding-top: 15px;
  border-top: 1px solid #eee;
}

.action-result h4 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #666;
}

.action-result pre {
  background: #f5f5f5;
  padding: 10px;
  border-radius: 4px;
  font-size: 12px;
  max-height: 150px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
