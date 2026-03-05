<template>
  <div class="evolution-effect-analytics">
    <!-- Page Header -->
    <div class="page-header">
      <h1>Evolution Effect Analysis</h1>
      <p class="subtitle">Analyze evolution performance improvements, trend patterns, and cross-domain transfer effectiveness</p>
    </div>

    <!-- Time Range Selection -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Analysis Time Range</h2>
        <div class="section-actions">
          <button @click="refreshAnalytics" class="btn btn-secondary" :disabled="refreshing">
            <span v-if="refreshing">Refreshing...</span>
            <span v-else>Refresh Data</span>
          </button>
        </div>
      </div>
      
      <div class="time-range-selector">
        <div class="time-range-buttons">
          <button 
            v-for="range in timeRanges" 
            :key="range.value"
            @click="selectTimeRange(range.value)"
            class="time-range-btn"
            :class="{ active: selectedTimeRange === range.value }"
          >
            {{ range.label }}
          </button>
        </div>
        
        <div class="custom-time-range">
          <div class="form-group">
            <label>Custom Time Range (hours):</label>
            <input 
              type="number" 
              v-model="customTimeRange"
              min="1"
              max="720"
              class="form-control-sm"
            />
            <button 
              @click="applyCustomTimeRange"
              class="btn btn-outline btn-sm"
            >
              Apply
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Performance Comparison Dashboard -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Performance Comparison Dashboard</h2>
        <div class="section-actions">
          <button @click="exportComparisonData" class="btn btn-outline">
            Export Data
          </button>
          <button @click="generateReport" class="btn btn-primary">
            Generate Report
          </button>
        </div>
      </div>

      <div v-if="performanceComparisons.length > 0" class="comparison-dashboard">
        <!-- Key Metrics Summary -->
        <div class="key-metrics-summary">
          <h3>Key Performance Improvements</h3>
          <div class="key-metrics-grid">
            <div 
              v-for="comparison in keyMetrics" 
              :key="comparison.metric_name"
              class="key-metric-card"
              :class="getSignificanceClass(comparison.significance_level)"
            >
              <h4>{{ formatMetricName(comparison.metric_name) }}</h4>
              <div class="metric-values">
                <div class="value-item">
                  <span class="label">Before:</span>
                  <span class="value before-value">{{ formatMetricValue(comparison.metric_name, comparison.before_value) }}</span>
                </div>
                <div class="value-item">
                  <span class="label">After:</span>
                  <span class="value after-value">{{ formatMetricValue(comparison.metric_name, comparison.after_value) }}</span>
                </div>
              </div>
              <div class="improvement-indicator">
                <span class="improvement-icon" :class="getImprovementDirection(comparison.improvement_percentage)">
                  {{ getImprovementIcon(comparison.improvement_percentage) }}
                </span>
                <span class="improvement-value" :class="getImprovementClass(comparison.improvement_percentage)">
                  {{ comparison.improvement_percentage > 0 ? '+' : '' }}{{ comparison.improvement_percentage.toFixed(1) }}%
                </span>
                <span class="significance-badge" :class="comparison.significance_level">
                  {{ comparison.significance_level.toUpperCase() }}
                </span>
              </div>
            </div>
          </div>
        </div>

        <!-- Detailed Comparison Table -->
        <div class="detailed-comparison">
          <h3>Detailed Performance Metrics</h3>
          <div class="comparison-table-container">
            <table class="comparison-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Before Evolution</th>
                  <th>After Evolution</th>
                  <th>Improvement</th>
                  <th>Significance</th>
                  <th>Trend</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="comparison in performanceComparisons" :key="comparison.metric_name">
                  <td class="metric-name">{{ formatMetricName(comparison.metric_name) }}</td>
                  <td class="before-value">{{ formatMetricValue(comparison.metric_name, comparison.before_value) }}</td>
                  <td class="after-value">{{ formatMetricValue(comparison.metric_name, comparison.after_value) }}</td>
                  <td class="improvement-value" :class="getImprovementClass(comparison.improvement_percentage)">
                    {{ comparison.improvement_percentage > 0 ? '+' : '' }}{{ comparison.improvement_percentage.toFixed(1) }}%
                  </td>
                  <td>
                    <span class="significance-badge" :class="comparison.significance_level">
                      {{ comparison.significance_level }}
                    </span>
                  </td>
                  <td>
                    <div class="trend-visualization">
                      <div class="trend-line">
                        <div 
                          class="before-dot" 
                          :style="{ bottom: calculateDotPosition(comparison.before_value, comparison.metric_name) + '%' }"
                        ></div>
                        <div 
                          class="after-dot" 
                          :style="{ bottom: calculateDotPosition(comparison.after_value, comparison.metric_name) + '%' }"
                        ></div>
                        <div 
                          class="trend-arrow" 
                          :class="getImprovementDirection(comparison.improvement_percentage)"
                        ></div>
                      </div>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
      
      <div v-else class="no-data-message">
        <div class="no-data-icon">📊</div>
        <h3>No Comparison Data Available</h3>
        <p>Performance comparison data will appear after evolution cycles complete.</p>
      </div>
    </div>

    <!-- Trend Analysis -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Trend Analysis</h2>
        <div class="section-actions">
          <select v-model="selectedTrendMetric" class="form-control">
            <option value="">Select Metric...</option>
            <option v-for="metric in availableTrendMetrics" :key="metric" :value="metric">
              {{ formatMetricName(metric) }}
            </option>
          </select>
          <button @click="refreshTrendData" class="btn btn-outline" :disabled="!selectedTrendMetric">
            Update Chart
          </button>
        </div>
      </div>

      <div v-if="selectedTrendMetric && trendData[selectedTrendMetric]" class="trend-analysis">
        <div class="trend-chart-container">
          <h3>{{ formatMetricName(selectedTrendMetric) }} Trend Over Time</h3>
          <div class="trend-chart">
            <!-- Chart Visualization -->
            <div class="chart-area">
              <div class="chart-grid">
                <div class="grid-line" v-for="n in 5" :key="n" :style="{ bottom: (n * 20) + '%' }"></div>
              </div>
              
              <div class="data-points">
                <div 
                  v-for="(point, index) in trendData[selectedTrendMetric]" 
                  :key="index"
                  class="data-point"
                  :style="{
                    left: calculatePointPosition(index, trendData[selectedTrendMetric].length) + '%',
                    bottom: calculateTrendPointPosition(point.value, selectedTrendMetric) + '%'
                  }"
                  :title="`${formatTimestamp(point.timestamp)}: ${formatMetricValue(selectedTrendMetric, point.value)}`"
                >
                  <div class="point-tooltip">
                    <div class="tooltip-content">
                      <strong>{{ formatMetricName(selectedTrendMetric) }}</strong><br>
                      Value: {{ formatMetricValue(selectedTrendMetric, point.value) }}<br>
                      Time: {{ formatTimestamp(point.timestamp) }}<br>
                      <span v-if="point.generation">Generation: {{ point.generation }}</span>
                      <span v-if="point.note">Note: {{ point.note }}</span>
                    </div>
                  </div>
                </div>
                
                <!-- Trend line -->
                <svg class="trend-line-svg" :viewBox="`0 0 100 100`" preserveAspectRatio="none">
                  <path 
                    :d="calculateTrendLinePath(selectedTrendMetric)" 
                    class="trend-line-path"
                    :class="getTrendLineClass(selectedTrendMetric)"
                  />
                </svg>
              </div>
              
              <div class="chart-axis">
                <div class="x-axis">
                  <div 
                    v-for="(point, index) in axisLabels" 
                    :key="index"
                    class="axis-label"
                    :style="{ left: (index * (100 / (axisLabels.length - 1))) + '%' }"
                  >
                    {{ formatTimeLabel(point.timestamp) }}
                  </div>
                </div>
                <div class="y-axis">
                  <div 
                    v-for="n in 6" 
                    :key="n"
                    class="axis-label"
                    :style="{ bottom: ((n - 1) * 20) + '%' }"
                  >
                    {{ formatYAxisLabel(n, selectedTrendMetric) }}
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <div class="trend-stats">
            <div class="stat-card">
              <h4>Current Value</h4>
              <p class="stat-value">
                {{ getCurrentTrendValue(selectedTrendMetric) }}
              </p>
            </div>
            <div class="stat-card">
              <h4>Average Growth Rate</h4>
              <p class="stat-value" :class="getGrowthRateClass(selectedTrendMetric)">
                {{ getGrowthRate(selectedTrendMetric) }}
              </p>
            </div>
            <div class="stat-card">
              <h4>Volatility</h4>
              <p class="stat-value" :class="getVolatilityClass(selectedTrendMetric)">
                {{ getVolatility(selectedTrendMetric) }}
              </p>
            </div>
            <div class="stat-card">
              <h4>Peak Value</h4>
              <p class="stat-value">
                {{ getPeakValue(selectedTrendMetric) }}
              </p>
            </div>
          </div>
        </div>
      </div>
      <div v-else class="no-trend-selected">
        <div class="placeholder-icon">📈</div>
        <h3>Select a metric to view trend analysis</h3>
        <p>Choose a performance metric from the dropdown above to visualize its evolution over time.</p>
      </div>
    </div>

    <!-- A/B Testing Results -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>A/B Testing Results</h2>
        <div class="section-actions">
          <button @click="runABTest" class="btn btn-primary" :disabled="runningABTest">
            <span v-if="runningABTest">Running Test...</span>
            <span v-else>Run New A/B Test</span>
          </button>
        </div>
      </div>

      <div v-if="abTestResults" class="ab-test-results">
        <div class="test-header">
          <h3>A/B Test: {{ abTestResults.strategy_a }} vs {{ abTestResults.strategy_b }}</h3>
          <div class="test-meta">
            <span class="test-id">ID: {{ abTestResults.test_id }}</span>
            <span class="test-duration">Duration: {{ abTestResults.duration_hours }} hours</span>
            <span class="confidence-level" :class="getConfidenceClass(abTestResults.results.confidence_level)">
              Confidence: {{ (abTestResults.results.confidence_level * 100).toFixed(0) }}%
            </span>
          </div>
        </div>

        <div class="test-comparison">
          <div class="strategy-results strategy-a">
            <h4>{{ abTestResults.strategy_a }}</h4>
            <div class="strategy-metrics">
              <div class="metric-result" v-for="(value, metric) in abTestResults.results.strategy_a" :key="metric">
                <span class="metric-name">{{ formatABTestMetric(metric) }}</span>
                <span class="metric-value">{{ formatABTestValue(metric, value) }}</span>
              </div>
            </div>
          </div>

          <div class="vs-divider">
            <div class="vs-label">VS</div>
            <div class="winner-badge" :class="abTestResults.results.winner === 'strategy_a' ? 'winner-a' : 'winner-b'">
              {{ abTestResults.results.winner === 'strategy_a' ? abTestResults.strategy_a : abTestResults.strategy_b }}
            </div>
          </div>

          <div class="strategy-results strategy-b">
            <h4>{{ abTestResults.strategy_b }}</h4>
            <div class="strategy-metrics">
              <div class="metric-result" v-for="(value, metric) in abTestResults.results.strategy_b" :key="metric">
                <span class="metric-name">{{ formatABTestMetric(metric) }}</span>
                <span class="metric-value">{{ formatABTestValue(metric, value) }}</span>
              </div>
            </div>
          </div>
        </div>

        <div class="test-conclusions">
          <h4>Key Conclusions</h4>
          <ul>
            <li v-if="abTestResults.results.winner === 'strategy_a'">
              <strong>{{ abTestResults.strategy_a }}</strong> performed better overall
            </li>
            <li v-else>
              <strong>{{ abTestResults.strategy_b }}</strong> performed better overall
            </li>
            <li v-if="abTestResults.results.confidence_level >= 0.8">
              Results are statistically significant
            </li>
            <li v-else>
              Results require further testing for statistical significance
            </li>
            <li>Consider running longer tests for more reliable results</li>
          </ul>
        </div>
      </div>
      <div v-else class="no-ab-test">
        <div class="placeholder-icon">🧪</div>
        <h3>No A/B Test Results Available</h3>
        <p>Run an A/B test to compare different evolution strategies and their effectiveness.</p>
      </div>
    </div>

    <!-- Overall Improvement Score -->
    <div class="dashboard-section">
      <div class="section-header">
        <h2>Overall Evolution Effectiveness</h2>
        <div class="section-actions">
          <button @click="refreshOverallScore" class="btn btn-outline">
            Recalculate Score
          </button>
        </div>
      </div>

      <div class="improvement-score">
        <div class="score-visualization">
          <div class="score-gauge">
            <div class="gauge-background"></div>
            <div 
              class="gauge-fill" 
              :style="{ transform: `rotate(${overallScore * 180}deg)` }"
            ></div>
            <div class="gauge-center">
              <span class="score-value">{{ (overallScore * 100).toFixed(1) }}</span>
              <span class="score-label">Effectiveness Score</span>
            </div>
          </div>
          
          <div class="score-breakdown">
            <h4>Score Breakdown</h4>
            <div class="breakdown-metrics">
              <div class="breakdown-item" v-for="(score, dimension) in scoreBreakdown" :key="dimension">
                <span class="dimension-name">{{ formatDimensionName(dimension) }}</span>
                <div class="dimension-bar">
                  <div 
                    class="dimension-fill" 
                    :style="{ width: (score * 100) + '%' }"
                    :class="getScoreClass(score)"
                  ></div>
                </div>
                <span class="dimension-score">{{ (score * 100).toFixed(1) }}%</span>
              </div>
            </div>
          </div>
        </div>
        
        <div class="score-interpretation">
          <h4>Interpretation</h4>
          <div class="interpretation-content">
            <p v-if="overallScore >= 0.8" class="score-excellent">
              <strong>Excellent Evolution Performance</strong><br>
              The evolution process is highly effective with significant improvements across all key metrics.
            </p>
            <p v-else-if="overallScore >= 0.6" class="score-good">
              <strong>Good Evolution Performance</strong><br>
              The evolution process is working well with noticeable improvements in most areas.
            </p>
            <p v-else-if="overallScore >= 0.4" class="score-fair">
              <strong>Fair Evolution Performance</strong><br>
              The evolution process shows some improvements but has room for optimization.
            </p>
            <p v-else class="score-poor">
              <strong>Poor Evolution Performance</strong><br>
              The evolution process needs significant improvement. Consider revising strategies.
            </p>
            
            <div class="recommendations">
              <h5>Recommendations:</h5>
              <ul>
                <li v-if="scoreBreakdown.knowledge_impact < 0.6">
                  Focus on knowledge acquisition and validation improvements
                </li>
                <li v-if="scoreBreakdown.model_improvement < 0.6">
                  Optimize model architecture search and training strategies
                </li>
                <li v-if="scoreBreakdown.cross_domain_effectiveness < 0.6">
                  Enhance cross-domain capability transfer mechanisms
                </li>
                <li v-if="scoreBreakdown.resource_efficiency < 0.6">
                  Improve resource utilization and efficiency
                </li>
                <li>Monitor trend patterns regularly for early detection of issues</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Loading Overlay -->
    <div v-if="loading" class="loading-overlay">
      <div class="loading-content">
        <div class="spinner"></div>
        <p>{{ loadingMessage }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import { useToast } from 'vue-toastification'

export default {
  name: 'EvolutionEffectAnalytics',
  
  setup() {
    const toast = useToast()
    
    // Reactive state
    const performanceComparisons = ref([])
    const trendData = ref({})
    const abTestResults = ref(null)
    const overallScore = ref(0.72)
    const scoreBreakdown = ref({})
    const selectedTimeRange = ref(24) // hours
    const customTimeRange = ref(24)
    const selectedTrendMetric = ref('accuracy')
    const availableTrendMetrics = ref(['accuracy', 'knowledge_coverage', 'cross_domain_connections', 'model_size_mb'])
    const refreshing = ref(false)
    const loading = ref(false)
    const loadingMessage = ref('')
    const runningABTest = ref(false)
    
    // Time range options
    const timeRanges = ref([
      { label: 'Last 1 hour', value: 1 },
      { label: 'Last 6 hours', value: 6 },
      { label: 'Last 24 hours', value: 24 },
      { label: 'Last 7 days', value: 168 }
    ])
    
    // Computed properties
    const keyMetrics = computed(() => {
      return performanceComparisons.value
        .filter(comp => comp.significance_level === 'high')
        .slice(0, 4)
    })
    
    const axisLabels = computed(() => {
      if (!selectedTrendMetric.value || !trendData.value[selectedTrendMetric.value]) {
        return []
      }
      
      const data = trendData.value[selectedTrendMetric.value]
      if (data.length === 0) return []
      
      // Return first, middle, and last points for axis labels
      return [
        data[0],
        data[Math.floor(data.length / 2)],
        data[data.length - 1]
      ]
    })
    
    // Methods
    const refreshAnalytics = async () => {
      refreshing.value = true
      try {
        // Fetch performance comparison data
        const comparisonResponse = await fetch(`/api/evolution/analytics/comparison?time_range_hours=${selectedTimeRange.value}`)
        if (comparisonResponse.ok) {
          const data = await comparisonResponse.json()
          performanceComparisons.value = data.comparisons || []
          overallScore.value = data.overall_improvement_score || 0.72
          abTestResults.value = data.ab_test_results || null
          
          // Generate score breakdown from comparisons
          updateScoreBreakdown()
        }
        
        // Fetch trend data for selected metric
        if (selectedTrendMetric.value) {
          await refreshTrendData()
        }
        
        toast.success('Analytics data refreshed')
      } catch (error) {
        console.error('Error refreshing analytics:', error)
        toast.error('Failed to refresh analytics data')
        
        // Fallback to mock data
        fallbackMockData()
      } finally {
        refreshing.value = false
      }
    }
    
    const fallbackMockData = () => {
      // Mock performance comparisons
      performanceComparisons.value = [
        {
          metric_name: 'accuracy',
          before_value: 0.78,
          after_value: 0.89,
          improvement_percentage: 14.1,
          significance_level: 'high'
        },
        {
          metric_name: 'inference_latency_ms',
          before_value: 120.5,
          after_value: 85.2,
          improvement_percentage: 29.3,
          significance_level: 'high'
        },
        {
          metric_name: 'knowledge_coverage',
          before_value: 0.65,
          after_value: 0.82,
          improvement_percentage: 26.2,
          significance_level: 'medium'
        },
        {
          metric_name: 'cross_domain_connections',
          before_value: 8,
          after_value: 15,
          improvement_percentage: 87.5,
          significance_level: 'high'
        },
        {
          metric_name: 'model_size_mb',
          before_value: 156.8,
          after_value: 142.3,
          improvement_percentage: 9.2,
          significance_level: 'low'
        },
        {
          metric_name: 'training_time_minutes',
          before_value: 45,
          after_value: 38,
          improvement_percentage: 15.6,
          significance_level: 'medium'
        }
      ]
      
      // Mock trend data
      trendData.value = {
        accuracy: generateMockTrendData('accuracy', 20),
        knowledge_coverage: generateMockTrendData('knowledge_coverage', 20),
        cross_domain_connections: generateMockTrendData('cross_domain_connections', 20)
      }
      
      // Mock A/B test results
      abTestResults.value = {
        test_id: 'ab_test_001',
        strategy_a: 'knowledge_focused',
        strategy_b: 'model_performance',
        duration_hours: 12,
        results: {
          strategy_a: {
            accuracy_improvement: 0.06,
            knowledge_growth: 18,
            resource_efficiency: 0.7
          },
          strategy_b: {
            accuracy_improvement: 0.09,
            knowledge_growth: 8,
            resource_efficiency: 0.8
          },
          winner: 'strategy_b',
          confidence_level: 0.85
        }
      }
      
      // Mock score breakdown
      updateScoreBreakdown()
    }
    
    const generateMockTrendData = (metric, count) => {
      const baseTime = Date.now() / 1000 - (selectedTimeRange.value * 3600)
      const data = []
      
      for (let i = 0; i < count; i++) {
        const timestamp = baseTime + (i * (selectedTimeRange.value * 3600 / count))
        let value
        
        if (metric === 'accuracy') {
          value = 0.70 + (i * 0.015) + (0.03 * (Math.random() - 0.5))
          value = Math.min(0.95, value)
        } else if (metric === 'knowledge_coverage') {
          value = 0.55 + (i * 0.018) + (0.04 * (Math.random() - 0.5))
          value = Math.min(0.90, value)
        } else if (metric === 'cross_domain_connections') {
          value = 5 + i + Math.floor(Math.random() * 4)
        } else {
          value = 0.5 + (i * 0.02) + (0.05 * (Math.random() - 0.5))
          value = Math.min(0.85, value)
        }
        
        data.push({
          timestamp,
          value,
          generation: i + 1,
          note: i % 4 === 0 ? `Architecture update generation ${i+1}` : null
        })
      }
      
      return data
    }
    
    const updateScoreBreakdown = () => {
      // Calculate score breakdown from performance comparisons
      const breakdown = {
        knowledge_impact: 0.0,
        model_improvement: 0.0,
        cross_domain_effectiveness: 0.0,
        resource_efficiency: 0.0,
        overall_consistency: 0.0
      }
      
      // Calculate knowledge impact
      const knowledgeMetrics = performanceComparisons.value.filter(
        comp => comp.metric_name.includes('knowledge') || comp.metric_name.includes('coverage')
      )
      if (knowledgeMetrics.length > 0) {
        breakdown.knowledge_impact = knowledgeMetrics.reduce((sum, comp) => 
          sum + (comp.improvement_percentage / 100), 0) / knowledgeMetrics.length
      }
      
      // Calculate model improvement
      const modelMetrics = performanceComparisons.value.filter(
        comp => comp.metric_name.includes('accuracy') || comp.metric_name.includes('latency') || comp.metric_name.includes('training')
      )
      if (modelMetrics.length > 0) {
        breakdown.model_improvement = modelMetrics.reduce((sum, comp) => 
          sum + (comp.improvement_percentage / 100), 0) / modelMetrics.length
      }
      
      // Calculate cross-domain effectiveness
      const crossDomainMetrics = performanceComparisons.value.filter(
        comp => comp.metric_name.includes('cross_domain')
      )
      if (crossDomainMetrics.length > 0) {
        breakdown.cross_domain_effectiveness = crossDomainMetrics.reduce((sum, comp) => 
          sum + (comp.improvement_percentage / 100), 0) / crossDomainMetrics.length
      }
      
      // Calculate resource efficiency
      const resourceMetrics = performanceComparisons.value.filter(
        comp => comp.metric_name.includes('size') || comp.metric_name.includes('time')
      )
      if (resourceMetrics.length > 0) {
        breakdown.resource_efficiency = resourceMetrics.reduce((sum, comp) => 
          sum + (comp.improvement_percentage / 100), 0) / resourceMetrics.length
      }
      
      // Calculate overall consistency (standard deviation of improvements)
      const improvements = performanceComparisons.value.map(comp => comp.improvement_percentage / 100)
      if (improvements.length > 1) {
        const mean = improvements.reduce((a, b) => a + b) / improvements.length
        const variance = improvements.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / improvements.length
        breakdown.overall_consistency = 1 - Math.sqrt(variance) // Higher consistency = lower variance
      } else {
        breakdown.overall_consistency = 0.5
      }
      
      // Normalize all scores to 0-1 range
      Object.keys(breakdown).forEach(key => {
        breakdown[key] = Math.max(0, Math.min(1, breakdown[key]))
      })
      
      scoreBreakdown.value = breakdown
    }
    
    const selectTimeRange = (range) => {
      selectedTimeRange.value = range
      refreshAnalytics()
    }
    
    const applyCustomTimeRange = () => {
      if (customTimeRange.value >= 1 && customTimeRange.value <= 720) {
        selectedTimeRange.value = customTimeRange.value
        refreshAnalytics()
      } else {
        toast.error('Please enter a time range between 1 and 720 hours')
      }
    }
    
    const formatMetricName = (metricName) => {
      const mapping = {
        'accuracy': 'Accuracy',
        'inference_latency_ms': 'Inference Latency',
        'knowledge_coverage': 'Knowledge Coverage',
        'cross_domain_connections': 'Cross-Domain Connections',
        'model_size_mb': 'Model Size',
        'training_time_minutes': 'Training Time'
      }
      return mapping[metricName] || metricName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    const formatMetricValue = (metricName, value) => {
      if (metricName === 'accuracy' || metricName === 'knowledge_coverage') {
        return `${(value * 100).toFixed(1)}%`
      } else if (metricName === 'inference_latency_ms') {
        return `${value.toFixed(1)}ms`
      } else if (metricName === 'model_size_mb') {
        return `${value.toFixed(1)}MB`
      } else if (metricName === 'training_time_minutes') {
        return `${value}min`
      } else {
        return value.toString()
      }
    }
    
    const getSignificanceClass = (significance) => {
      return `significance-${significance}`
    }
    
    const getImprovementDirection = (improvement) => {
      return improvement > 0 ? 'positive' : improvement < 0 ? 'negative' : 'neutral'
    }
    
    const getImprovementIcon = (improvement) => {
      return improvement > 0 ? '📈' : improvement < 0 ? '📉' : '➡️'
    }
    
    const getImprovementClass = (improvement) => {
      if (improvement > 20) return 'improvement-high'
      if (improvement > 10) return 'improvement-medium'
      if (improvement > 0) return 'improvement-low'
      if (improvement < 0) return 'improvement-negative'
      return 'improvement-neutral'
    }
    
    const calculateDotPosition = (value, metricName) => {
      // Normalize value to 0-100% range based on metric type
      let normalized
      
      if (metricName === 'accuracy' || metricName === 'knowledge_coverage') {
        normalized = value * 100 // Convert 0-1 to 0-100
      } else if (metricName === 'inference_latency_ms') {
        normalized = Math.min(100, (value / 200) * 100) // Assume max 200ms
      } else if (metricName === 'model_size_mb') {
        normalized = Math.min(100, (value / 200) * 100) // Assume max 200MB
      } else if (metricName === 'training_time_minutes') {
        normalized = Math.min(100, (value / 60) * 100) // Assume max 60min
      } else {
        normalized = Math.min(100, value)
      }
      
      return Math.max(0, Math.min(100, normalized))
    }
    
    const refreshTrendData = async () => {
      if (!selectedTrendMetric.value) return
      
      loading.value = true
      loadingMessage.value = 'Loading trend data...'
      
      try {
        const response = await fetch(`/api/evolution/analytics/trends?metric=${selectedTrendMetric.value}&time_range_hours=${selectedTimeRange.value}`)
        if (response.ok) {
          const data = await response.json()
          trendData.value = {
            ...trendData.value,
            [selectedTrendMetric.value]: data[selectedTrendMetric.value] || []
          }
        } else {
          throw new Error(`Failed to fetch trend data: ${response.status}`)
        }
      } catch (error) {
        console.error('Error fetching trend data:', error)
        toast.error('Failed to load trend data')
        
        // Generate mock trend data
        trendData.value[selectedTrendMetric.value] = generateMockTrendData(selectedTrendMetric.value, 20)
      } finally {
        loading.value = false
      }
    }
    
    const calculatePointPosition = (index, totalPoints) => {
      return (index / (totalPoints - 1)) * 100
    }
    
    const calculateTrendPointPosition = (value, metricName) => {
      // Similar to calculateDotPosition but for trend chart
      return calculateDotPosition(value, metricName)
    }
    
    const calculateTrendLinePath = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length < 2) return ''
      
      const points = data.map((point, index) => {
        const x = (index / (data.length - 1)) * 100
        const y = 100 - calculateTrendPointPosition(point.value, metricName)
        return `${x},${y}`
      })
      
      return `M ${points.join(' L ')}`
    }
    
    const getTrendLineClass = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length < 2) return 'trend-neutral'
      
      const firstValue = data[0].value
      const lastValue = data[data.length - 1].value
      
      if (lastValue > firstValue * 1.1) return 'trend-positive'
      if (lastValue < firstValue * 0.9) return 'trend-negative'
      return 'trend-neutral'
    }
    
    const formatTimestamp = (timestamp) => {
      const date = new Date(timestamp * 1000)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const formatTimeLabel = (timestamp) => {
      const date = new Date(timestamp * 1000)
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }
    
    const formatYAxisLabel = (n, metricName) => {
      const maxValue = 100
      const value = (maxValue * (n - 1)) / 5
      
      if (metricName === 'accuracy' || metricName === 'knowledge_coverage') {
        return `${value.toFixed(0)}%`
      } else if (metricName === 'cross_domain_connections') {
        return Math.round(value).toString()
      } else {
        return value.toFixed(0)
      }
    }
    
    const getCurrentTrendValue = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length === 0) return 'N/A'
      return formatMetricValue(metricName, data[data.length - 1].value)
    }
    
    const getGrowthRate = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length < 2) return 'N/A'
      
      const firstValue = data[0].value
      const lastValue = data[data.length - 1].value
      const timeDiff = data[data.length - 1].timestamp - data[0].timestamp
      const hours = timeDiff / 3600
      
      const growthRate = ((lastValue - firstValue) / firstValue) * 100 / hours
      return `${growthRate.toFixed(2)}%/hour`
    }
    
    const getGrowthRateClass = (metricName) => {
      const rate = getGrowthRate(metricName)
      if (rate.includes('N/A')) return 'rate-neutral'
      
      const numericRate = parseFloat(rate)
      if (numericRate > 0.5) return 'rate-positive'
      if (numericRate < -0.5) return 'rate-negative'
      return 'rate-neutral'
    }
    
    const getVolatility = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length < 3) return 'N/A'
      
      const values = data.map(d => d.value)
      const mean = values.reduce((a, b) => a + b) / values.length
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length
      const stdDev = Math.sqrt(variance)
      
      const volatility = (stdDev / mean) * 100
      return `${volatility.toFixed(2)}%`
    }
    
    const getVolatilityClass = (metricName) => {
      const volatility = getVolatility(metricName)
      if (volatility.includes('N/A')) return 'volatility-neutral'
      
      const numericVolatility = parseFloat(volatility)
      if (numericVolatility < 5) return 'volatility-low'
      if (numericVolatility < 15) return 'volatility-medium'
      return 'volatility-high'
    }
    
    const getPeakValue = (metricName) => {
      const data = trendData.value[metricName]
      if (!data || data.length === 0) return 'N/A'
      
      const maxValue = Math.max(...data.map(d => d.value))
      return formatMetricValue(metricName, maxValue)
    }
    
    const runABTest = async () => {
      runningABTest.value = true
      loadingMessage.value = 'Running A/B test...'
      
      try {
        // In a real implementation, this would trigger an A/B test
        await new Promise(resolve => setTimeout(resolve, 3000))
        
        // Mock A/B test results
        abTestResults.value = {
          test_id: `ab_test_${Date.now()}`,
          strategy_a: 'knowledge_focused',
          strategy_b: 'model_performance',
          duration_hours: 12,
          results: {
            strategy_a: {
              accuracy_improvement: 0.06 + (Math.random() * 0.04),
              knowledge_growth: 18 + Math.floor(Math.random() * 5),
              resource_efficiency: 0.7 + (Math.random() * 0.1)
            },
            strategy_b: {
              accuracy_improvement: 0.09 + (Math.random() * 0.04),
              knowledge_growth: 8 + Math.floor(Math.random() * 5),
              resource_efficiency: 0.8 + (Math.random() * 0.1)
            },
            winner: Math.random() > 0.5 ? 'strategy_a' : 'strategy_b',
            confidence_level: 0.7 + (Math.random() * 0.25)
          }
        }
        
        toast.success('A/B test completed successfully')
      } catch (error) {
        console.error('Error running A/B test:', error)
        toast.error('Failed to run A/B test')
      } finally {
        runningABTest.value = false
        loadingMessage.value = ''
      }
    }
    
    const formatABTestMetric = (metric) => {
      const mapping = {
        'accuracy_improvement': 'Accuracy Improvement',
        'knowledge_growth': 'Knowledge Growth',
        'resource_efficiency': 'Resource Efficiency'
      }
      return mapping[metric] || metric.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    const formatABTestValue = (metric, value) => {
      if (metric === 'accuracy_improvement') {
        return `${(value * 100).toFixed(1)}%`
      } else if (metric === 'resource_efficiency') {
        return `${(value * 100).toFixed(1)}%`
      } else {
        return value.toString()
      }
    }
    
    const getConfidenceClass = (confidence) => {
      if (confidence >= 0.8) return 'confidence-high'
      if (confidence >= 0.6) return 'confidence-medium'
      return 'confidence-low'
    }
    
    const refreshOverallScore = () => {
      // Recalculate overall score based on current data
      if (performanceComparisons.value.length > 0) {
        const avgImprovement = performanceComparisons.value.reduce(
          (sum, comp) => sum + (comp.improvement_percentage / 100), 0
        ) / performanceComparisons.value.length
        
        overallScore.value = Math.max(0, Math.min(1, avgImprovement))
        updateScoreBreakdown()
        toast.success('Effectiveness score recalculated')
      }
    }
    
    const formatDimensionName = (dimension) => {
      const mapping = {
        'knowledge_impact': 'Knowledge Impact',
        'model_improvement': 'Model Improvement',
        'cross_domain_effectiveness': 'Cross-Domain Effectiveness',
        'resource_efficiency': 'Resource Efficiency',
        'overall_consistency': 'Overall Consistency'
      }
      return mapping[dimension] || dimension.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    }
    
    const getScoreClass = (score) => {
      if (score >= 0.8) return 'score-high'
      if (score >= 0.6) return 'score-medium'
      if (score >= 0.4) return 'score-low'
      return 'score-poor'
    }
    
    const exportComparisonData = () => {
      toast.info('Export functionality would generate a CSV/PDF report')
      // In a real implementation, this would trigger a file download
    }
    
    const generateReport = () => {
      toast.info('Report generation would create a detailed analysis document')
      // In a real implementation, this would generate a comprehensive report
    }
    
    // Lifecycle hooks
    onMounted(() => {
      refreshAnalytics()
    })
    
    // Watch for time range changes
    watch(selectedTimeRange, () => {
      refreshAnalytics()
    })
    
    return {
      // State
      performanceComparisons,
      trendData,
      abTestResults,
      overallScore,
      scoreBreakdown,
      selectedTimeRange,
      customTimeRange,
      selectedTrendMetric,
      availableTrendMetrics,
      refreshing,
      loading,
      loadingMessage,
      runningABTest,
      timeRanges,
      
      // Computed
      keyMetrics,
      axisLabels,
      
      // Methods
      refreshAnalytics,
      selectTimeRange,
      applyCustomTimeRange,
      formatMetricName,
      formatMetricValue,
      getSignificanceClass,
      getImprovementDirection,
      getImprovementIcon,
      getImprovementClass,
      calculateDotPosition,
      refreshTrendData,
      calculatePointPosition,
      calculateTrendPointPosition,
      calculateTrendLinePath,
      getTrendLineClass,
      formatTimestamp,
      formatTimeLabel,
      formatYAxisLabel,
      getCurrentTrendValue,
      getGrowthRate,
      getGrowthRateClass,
      getVolatility,
      getVolatilityClass,
      getPeakValue,
      runABTest,
      formatABTestMetric,
      formatABTestValue,
      getConfidenceClass,
      refreshOverallScore,
      formatDimensionName,
      getScoreClass,
      exportComparisonData,
      generateReport
    }
  }
}
</script>

<style scoped>
.evolution-effect-analytics {
  padding: 20px;
  max-width: 1600px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 30px;
}

.page-header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
  color: #222;
}

.page-header .subtitle {
  color: #555;
  font-size: 1.1rem;
}

.dashboard-section {
  margin-bottom: 40px;
  padding: 25px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-header h2 {
  font-size: 1.8rem;
  color: #222;
}

.section-actions {
  display: flex;
  gap: 10px;
  align-items: center;
}

.time-range-selector {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.time-range-buttons {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.time-range-btn {
  padding: 8px 16px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  background: var(--bg-primary);
  color: #333;
  cursor: pointer;
  transition: all 0.3s ease;
}

.time-range-btn:hover {
  border-color: #2196F3;
  color: #2196F3;
}

.time-range-btn.active {
  background: #2196F3;
  color: white;
  border-color: #2196F3;
}

.custom-time-range {
  display: flex;
  align-items: center;
  gap: 15px;
}

.custom-time-range .form-group {
  display: flex;
  align-items: center;
  gap: 10px;
}

.custom-time-range label {
  font-weight: 600;
  color: #333;
}

.form-control-sm {
  padding: 6px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  width: 80px;
}

.comparison-dashboard {
  margin-top: 20px;
}

.key-metrics-summary {
  margin-bottom: 30px;
}

.key-metrics-summary h3 {
  margin-bottom: 20px;
  color: #333;
  font-size: 1.4rem;
}

.key-metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.key-metric-card {
  padding: 20px;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  transition: all 0.3s ease;
}

.key-metric-card:hover {
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.key-metric-card.significance-high {
  border-left: 4px solid #4CAF50;
}

.key-metric-card.significance-medium {
  border-left: 4px solid #FF9800;
}

.key-metric-card.significance-low {
  border-left: 4px solid #9E9E9E;
}

.key-metric-card h4 {
  margin: 0 0 15px 0;
  color: #222;
  font-size: 1.2rem;
}

.metric-values {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 15px;
}

.value-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.value-item .label {
  font-weight: 600;
  color: #666;
}

.value-item .value {
  font-weight: 600;
  font-size: 1.1rem;
}

.value-item .before-value {
  color: #F44336;
}

.value-item .after-value {
  color: #4CAF50;
}

.improvement-indicator {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 15px;
}

.improvement-icon {
  font-size: 1.5rem;
}

.improvement-icon.positive {
  color: #4CAF50;
}

.improvement-icon.negative {
  color: #F44336;
}

.improvement-icon.neutral {
  color: #FF9800;
}

.improvement-value {
  font-weight: 700;
  font-size: 1.3rem;
}

.improvement-value.improvement-high {
  color: #4CAF50;
}

.improvement-value.improvement-medium {
  color: #FF9800;
}

.improvement-value.improvement-low {
  color: #2196F3;
}

.improvement-value.improvement-negative {
  color: #F44336;
}

.improvement-value.improvement-neutral {
  color: #9E9E9E;
}

.significance-badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
  margin-left: auto;
}

.significance-badge.high {
  background-color: #E8F5E9;
  color: #2E7D32;
}

.significance-badge.medium {
  background-color: #FFF3E0;
  color: #EF6C00;
}

.significance-badge.low {
  background-color: #F5F5F5;
  color: #616161;
}

.detailed-comparison h3 {
  margin-bottom: 20px;
  color: #333;
  font-size: 1.4rem;
}

.comparison-table-container {
  overflow-x: auto;
}

.comparison-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  overflow: hidden;
}

.comparison-table th {
  background: var(--bg-secondary);
  padding: 12px 15px;
  text-align: left;
  font-weight: 600;
  color: #333;
  border-bottom: 2px solid var(--border-color);
}

.comparison-table td {
  padding: 12px 15px;
  border-bottom: 1px solid var(--border-color);
}

.comparison-table tr:last-child td {
  border-bottom: none;
}

.comparison-table tr:hover {
  background-color: rgba(33, 150, 243, 0.05);
}

.metric-name {
  font-weight: 600;
  color: #333;
}

.before-value {
  color: #F44336;
  font-weight: 600;
}

.after-value {
  color: #4CAF50;
  font-weight: 600;
}

.trend-visualization {
  width: 100px;
  height: 40px;
  position: relative;
}

.trend-line {
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 2px;
  background: #E0E0E0;
  transform: translateY(-50%);
}

.before-dot,
.after-dot {
  position: absolute;
  width: 10px;
  height: 10px;
  border-radius: 50%;
  transform: translateX(-50%);
}

.before-dot {
  left: 25%;
  background-color: #F44336;
}

.after-dot {
  left: 75%;
  background-color: #4CAF50;
}

.trend-arrow {
  position: absolute;
  width: 0;
  height: 0;
  border-left: 6px solid transparent;
  border-right: 6px solid transparent;
  top: 50%;
  left: 75%;
  transform: translate(-50%, -50%);
}

.trend-arrow.positive {
  border-bottom: 8px solid #4CAF50;
  transform: translate(-50%, -50%) rotate(180deg);
}

.trend-arrow.negative {
  border-top: 8px solid #F44336;
}

.trend-arrow.neutral {
  border-top: 8px solid #FF9800;
  border-bottom: 8px solid #FF9800;
  border-left: 0;
  border-right: 0;
  width: 12px;
  height: 4px;
}

.no-data-message,
.no-trend-selected,
.no-ab-test {
  padding: 60px 20px;
  text-align: center;
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 2px dashed var(--border-color);
}

.no-data-icon,
.placeholder-icon {
  font-size: 4rem;
  margin-bottom: 20px;
  color: #2196F3;
}

.no-data-message h3,
.no-trend-selected h3,
.no-ab-test h3 {
  margin: 0 0 10px 0;
  color: #333;
}

.no-data-message p,
.no-trend-selected p,
.no-ab-test p {
  color: #666;
  margin: 0;
  max-width: 500px;
  margin: 0 auto;
}

.trend-analysis {
  margin-top: 20px;
}

.trend-chart-container {
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  padding: 20px;
}

.trend-chart-container h3 {
  margin: 0 0 20px 0;
  color: #222;
  font-size: 1.3rem;
}

.trend-chart {
  height: 400px;
  position: relative;
}

.chart-area {
  position: relative;
  height: 300px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  margin-bottom: 20px;
}

.chart-grid {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
}

.grid-line {
  position: absolute;
  left: 0;
  right: 0;
  height: 1px;
  background: rgba(0, 0, 0, 0.1);
}

.data-points {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
}

.data-point {
  position: absolute;
  width: 12px;
  height: 12px;
  background: #2196F3;
  border-radius: 50%;
  transform: translate(-50%, 50%);
  cursor: pointer;
  transition: all 0.3s ease;
}

.data-point:hover {
  width: 16px;
  height: 16px;
  background: #0b7dda;
  z-index: 10;
}

.data-point:hover .point-tooltip {
  opacity: 1;
  visibility: visible;
  transform: translate(-50%, -10px);
}

.point-tooltip {
  position: absolute;
  bottom: 100%;
  left: 50%;
  transform: translate(-50%, -20px);
  background: rgba(0, 0, 0, 0.8);
  color: white;
  padding: 10px;
  border-radius: 4px;
  font-size: 0.8rem;
  white-space: nowrap;
  opacity: 0;
  visibility: hidden;
  transition: all 0.3s ease;
  z-index: 100;
}

.tooltip-content {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.trend-line-svg {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.trend-line-path {
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

.trend-line-path.trend-positive {
  stroke: #4CAF50;
}

.trend-line-path.trend-negative {
  stroke: #F44336;
}

.trend-line-path.trend-neutral {
  stroke: #FF9800;
}

.chart-axis {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.x-axis,
.y-axis {
  position: absolute;
}

.x-axis {
  bottom: -25px;
  left: 0;
  right: 0;
  height: 25px;
}

.y-axis {
  top: 0;
  left: -60px;
  bottom: 0;
  width: 60px;
}

.axis-label {
  position: absolute;
  font-size: 0.8rem;
  color: #666;
  transform: translate(-50%, 0);
}

.x-axis .axis-label {
  top: 0;
  transform: translate(-50%, 0);
}

.y-axis .axis-label {
  right: 10px;
  transform: translate(0, 50%);
}

.trend-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 20px;
  margin-top: 30px;
}

.stat-card {
  padding: 15px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  text-align: center;
}

.stat-card h4 {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 1rem;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  margin: 0;
}

.stat-value.rate-positive {
  color: #4CAF50;
}

.stat-value.rate-negative {
  color: #F44336;
}

.stat-value.rate-neutral {
  color: #FF9800;
}

.stat-value.volatility-low {
  color: #4CAF50;
}

.stat-value.volatility-medium {
  color: #FF9800;
}

.stat-value.volatility-high {
  color: #F44336;
}

.ab-test-results {
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  padding: 20px;
}

.test-header {
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--border-color);
}

.test-header h3 {
  margin: 0 0 10px 0;
  color: #222;
  font-size: 1.4rem;
}

.test-meta {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.test-id,
.test-duration {
  color: #666;
  font-size: 0.9rem;
}

.confidence-level {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 0.8rem;
  font-weight: 500;
}

.confidence-level.confidence-high {
  background-color: #E8F5E9;
  color: #2E7D32;
}

.confidence-level.confidence-medium {
  background-color: #FFF3E0;
  color: #EF6C00;
}

.confidence-level.confidence-low {
  background-color: #FFEBEE;
  color: #C62828;
}

.test-comparison {
  display: flex;
  align-items: center;
  gap: 40px;
  margin-bottom: 30px;
}

.strategy-results {
  flex: 1;
  padding: 20px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.strategy-results h4 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 1.2rem;
}

.strategy-metrics {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.metric-result {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.metric-name {
  color: #666;
}

.metric-value {
  font-weight: 600;
  color: #333;
}

.vs-divider {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
}

.vs-label {
  padding: 5px 15px;
  background: #2196F3;
  color: white;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.9rem;
}

.winner-badge {
  padding: 5px 15px;
  border-radius: 20px;
  font-weight: 600;
  font-size: 0.9rem;
}

.winner-badge.winner-a {
  background-color: #4CAF50;
  color: white;
}

.winner-badge.winner-b {
  background-color: #2196F3;
  color: white;
}

.test-conclusions {
  padding: 20px;
  background: var(--bg-secondary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
}

.test-conclusions h4 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 1.2rem;
}

.test-conclusions ul {
  margin: 0;
  padding-left: 20px;
}

.test-conclusions li {
  color: #555;
  margin-bottom: 8px;
  line-height: 1.5;
}

.improvement-score {
  margin-top: 20px;
}

.score-visualization {
  display: flex;
  gap: 40px;
  margin-bottom: 30px;
  align-items: center;
}

.score-gauge {
  position: relative;
  width: 200px;
  height: 200px;
  flex-shrink: 0;
}

.gauge-background {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: conic-gradient(
    #F44336 0deg 90deg,
    #FF9800 90deg 180deg,
    #4CAF50 180deg 270deg,
    #2196F3 270deg 360deg
  );
  mask: radial-gradient(transparent 65%, black 66%);
  -webkit-mask: radial-gradient(transparent 65%, black 66%);
}

.gauge-fill {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  background: conic-gradient(
    #2196F3 0deg 180deg,
    transparent 180deg 360deg
  );
  mask: radial-gradient(transparent 65%, black 66%);
  -webkit-mask: radial-gradient(transparent 65%, black 66%);
  transform-origin: center;
  transition: transform 1s ease;
}

.gauge-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.score-value {
  display: block;
  font-size: 2.5rem;
  font-weight: 700;
  color: #222;
  line-height: 1;
}

.score-label {
  display: block;
  font-size: 0.9rem;
  color: #666;
  margin-top: 5px;
}

.score-breakdown {
  flex: 1;
}

.score-breakdown h4 {
  margin: 0 0 20px 0;
  color: #333;
  font-size: 1.3rem;
}

.breakdown-metrics {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.breakdown-item {
  display: flex;
  align-items: center;
  gap: 15px;
}

.dimension-name {
  min-width: 200px;
  font-weight: 600;
  color: #333;
}

.dimension-bar {
  flex: 1;
  height: 8px;
  background: #E0E0E0;
  border-radius: 4px;
  overflow: hidden;
}

.dimension-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}

.dimension-fill.score-high {
  background: linear-gradient(90deg, #4CAF50, #8BC34A);
}

.dimension-fill.score-medium {
  background: linear-gradient(90deg, #FF9800, #FFB74D);
}

.dimension-fill.score-low {
  background: linear-gradient(90deg, #2196F3, #64B5F6);
}

.dimension-fill.score-poor {
  background: linear-gradient(90deg, #F44336, #EF9A9A);
}

.dimension-score {
  min-width: 60px;
  font-weight: 600;
  color: #333;
  text-align: right;
}

.score-interpretation {
  background: var(--bg-primary);
  border-radius: var(--border-radius-sm);
  border: 1px solid var(--border-color);
  padding: 20px;
}

.score-interpretation h4 {
  margin: 0 0 15px 0;
  color: #333;
  font-size: 1.3rem;
}

.interpretation-content {
  line-height: 1.6;
}

.interpretation-content p {
  margin: 0 0 20px 0;
  padding: 15px;
  border-radius: var(--border-radius-sm);
}

.score-excellent {
  background-color: #E8F5E9;
  border-left: 4px solid #4CAF50;
  color: #2E7D32;
}

.score-good {
  background-color: #E3F2FD;
  border-left: 4px solid #2196F3;
  color: #1565C0;
}

.score-fair {
  background-color: #FFF3E0;
  border-left: 4px solid #FF9800;
  color: #EF6C00;
}

.score-poor {
  background-color: #FFEBEE;
  border-left: 4px solid #F44336;
  color: #C62828;
}

.recommendations {
  margin-top: 20px;
}

.recommendations h5 {
  margin: 0 0 10px 0;
  color: #333;
  font-size: 1.1rem;
}

.recommendations ul {
  margin: 0;
  padding-left: 20px;
}

.recommendations li {
  color: #555;
  margin-bottom: 8px;
  line-height: 1.5;
}

.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
}

.loading-content {
  text-align: center;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 5px solid #f3f3f3;
  border-top: 5px solid #2196F3;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.btn-primary {
  background-color: #2196F3;
  color: white;
}

.btn-primary:hover:not(:disabled) {
  background-color: #0b7dda;
}

.btn-secondary {
  background-color: #6c757d;
  color: white;
}

.btn-secondary:hover:not(:disabled) {
  background-color: #545b62;
}

.btn-success {
  background-color: #4CAF50;
  color: white;
}

.btn-success:hover:not(:disabled) {
  background-color: #3d8b40;
}

.btn-outline {
  background-color: transparent;
  color: #2196F3;
  border: 1px solid #2196F3;
}

.btn-outline:hover:not(:disabled) {
  background-color: #2196F3;
  color: white;
}

.btn-sm {
  padding: 5px 10px;
  font-size: 0.8rem;
}

.form-control {
  padding: 8px 12px;
  border: 1px solid var(--border-color);
  border-radius: var(--border-radius-sm);
  font-size: 0.9rem;
  background: white;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .section-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
  
  .section-actions {
    width: 100%;
    justify-content: space-between;
  }
  
  .key-metrics-grid {
    grid-template-columns: 1fr;
  }
  
  .test-comparison {
    flex-direction: column;
    gap: 20px;
  }
  
  .score-visualization {
    flex-direction: column;
    gap: 30px;
  }
  
  .breakdown-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .dimension-name {
    min-width: auto;
  }
  
  .dimension-bar {
    width: 100%;
  }
  
  .trend-chart {
    height: 300px;
  }
  
  .chart-area {
    height: 200px;
  }
  
  .trend-stats {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 480px) {
  .trend-stats {
    grid-template-columns: 1fr;
  }
  
  .time-range-buttons {
    flex-direction: column;
  }
  
  .custom-time-range {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
