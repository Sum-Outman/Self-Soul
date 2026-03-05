<template>
  <div class="goal-metrics">
    <h2>Goal Metrics Dashboard</h2>
    <div v-if="loading" class="loading">Loading goal data...</div>
    <div v-else-if="error" class="error">Error loading goals: {{ error }}</div>
    <div v-else class="metrics-grid">
      <!-- Overall goal metrics -->
      <metric-card
        title="Total Goals"
        :value="goalReport.total_goals"
        type="count"
        icon="fa-bullseye"
        color="#4CAF50"
        trend="stable"
      ></metric-card>
      
      <metric-card
        title="Average Progress"
        :value="goalReport.average_progress"
        type="progress"
        :max="100"
        icon="fa-chart-line"
        color="#2196F3"
        :trend="getProgressTrend(goalReport.average_progress)"
      ></metric-card>
      
      <metric-card
        title="Critical Goals"
        :value="criticalGoals.count"
        type="count"
        icon="fa-exclamation-triangle"
        color="#FF5722"
        trend="up"
      ></metric-card>
      
      <metric-card
        title="Goal Completion"
        :value="calculateCompletionRate()"
        type="progress"
        :max="100"
        icon="fa-check-circle"
        color="#9C27B0"
        trend="stable"
      ></metric-card>
    </div>
    
    <!-- Critical goals list -->
    <div v-if="criticalGoals.count > 0" class="critical-goals-section">
      <h3>Critical Goals Requiring Attention</h3>
      <div class="critical-goals-list">
        <div v-for="goal in criticalGoals.critical_goals" :key="goal.goal_id" class="critical-goal-item">
          <h4>{{ goal.description }}</h4>
          <p>Progress: {{ (goal.progress * 100).toFixed(1) }}%</p>
          <p>Priority: {{ goal.priority }}</p>
          <p v-if="goal.deadline">Deadline: {{ formatDate(goal.deadline) }}</p>
          <p>Suggested Actions: {{ goal.suggested_actions.join(', ') }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { apiClient } from '@/utils/api';
import MetricCard from './MetricCard.vue';

export default {
  name: 'GoalMetrics',
  components: {
    MetricCard
  },
  data() {
    return {
      loading: true,
      error: null,
      goalReport: {
        total_goals: 0,
        average_progress: 0,
        priority_distribution: {},
        goals_with_valid_progress: 0,
        last_updated: ''
      },
      criticalGoals: {
        count: 0,
        critical_goals: []
      }
    };
  },
  mounted() {
    this.fetchGoalData();
  },
  methods: {
    async fetchGoalData() {
      this.loading = true;
      this.error = null;
      
      try {
        // Fetch both goal report and critical goals in parallel
        const [reportResponse, criticalResponse] = await Promise.all([
          apiClient.get('/api/goals'),
          apiClient.get('/api/goals/critical')
        ]);
        
        if (reportResponse.data.status === 'success') {
          this.goalReport = reportResponse.data.data;
        }
        
        if (criticalResponse.data.status === 'success') {
          this.criticalGoals = criticalResponse.data.data;
        }
      } catch (err) {
        console.error('Failed to fetch goal data:', err);
        this.error = err.message || 'Unknown error occurred';
      } finally {
        this.loading = false;
      }
    },
    
    getProgressTrend(progress) {
      if (progress < 30) return 'down';
      if (progress > 70) return 'up';
      return 'stable';
    },
    
    calculateCompletionRate() {
      if (this.goalReport.total_goals === 0) return 0;
      const completedGoals = Object.values(this.goalReport.priority_distribution || {})
        .reduce((sum, count) => sum + count, 0);
      return (completedGoals / this.goalReport.total_goals) * 100;
    },
    
    formatDate(dateString) {
      if (!dateString) return 'N/A';
      const date = new Date(dateString);
      return date.toLocaleDateString();
    }
  }
};
</script>

<style scoped>
.goal-metrics {
  padding: 20px;
  background: #f5f5f5;
  border-radius: 8px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 30px;
}

.loading, .error {
  padding: 20px;
  text-align: center;
  font-size: 18px;
}

.error {
  color: #ff5722;
}

.critical-goals-section {
  margin-top: 30px;
  padding: 20px;
  background: #fff3cd;
  border-radius: 8px;
  border-left: 5px solid #ffc107;
}

.critical-goals-list {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.critical-goal-item {
  padding: 15px;
  background: white;
  border-radius: 6px;
  border-left: 4px solid #ff5722;
}

.critical-goal-item h4 {
  margin: 0 0 10px 0;
  color: #333;
}

.critical-goal-item p {
  margin: 5px 0;
  color: #666;
}
</style>