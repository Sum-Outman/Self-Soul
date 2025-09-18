<template>
  <div class="help-container">

    <!-- Main Content -->
    <main class="help-main">
      <!-- System Introduction -->
      <section class="help-section">
        <h2>系统介绍</h2>
        <p>Self Soul 是一个革命性的类人脑AGI系统，具备独立思考、自主学习、自我优化和自我升级的能力。系统采用多模态架构，整合了语言、视觉、音频、传感器等多种感知能力。</p>
      </section>

      <!-- Quick Start -->
      <section class="help-section">
        <h2>快速开始</h2>
        <div class="step-list">
          <div class="step">
            <span class="step-number">1</span>
            <div class="step-content">
              <h3>启动系统</h3>
              <p>运行启动脚本启动AGI系统</p>
            </div>
          </div>
          <div class="step">
            <span class="step-number">2</span>
            <div class="step-content">
              <h3>配置模型</h3>
              <p>在设置页面配置本地或外部API模型</p>
            </div>
          </div>
          <div class="step">
            <span class="step-number">3</span>
            <div class="step-content">
              <h3>开始使用</h3>
              <p>通过交互界面与系统进行多模态对话</p>
            </div>
          </div>
        </div>
      </section>

      <!-- Core Models -->
      <section class="help-section">
        <h2>核心模型</h2>
        <div class="model-grid">
          <div class="model-card">
            <h3>管理模型</h3>
            <p>负责任务分配和模型协调</p>
          </div>
          <div class="model-card">
            <h3>语言模型</h3>
            <p>处理文本交互和对话</p>
          </div>
          <div class="model-card">
            <h3>视觉模型</h3>
            <p>处理图像和视频内容</p>
          </div>
          <div class="model-card">
            <h3>音频模型</h3>
            <p>处理语音和音频输入</p>
          </div>
          <div class="model-card">
            <h3>自主模型</h3>
            <p>实现系统自主决策和行动</p>
          </div>
          <div class="model-card">
            <h3>知识模型</h3>
            <p>管理和处理知识库内容</p>
          </div>
          <div class="model-card">
            <h3>规划模型</h3>
            <p>制定任务计划和执行策略</p>
          </div>
          <div class="model-card">
            <h3>预测模型</h3>
            <p>进行数据分析和趋势预测</p>
          </div>
          <div class="model-card">
            <h3>编程模型</h3>
            <p>生成和优化代码</p>
          </div>
          <div class="model-card">
            <h3>传感器模型</h3>
            <p>处理各种传感器输入数据</p>
          </div>
          <div class="model-card">
            <h3>空间模型</h3>
            <p>理解和处理空间关系</p>
          </div>
          <div class="model-card">
            <h3>情感模型</h3>
            <p>识别和处理情感信息</p>
          </div>
        </div>
      </section>

      <!-- Training Guide -->
      <section class="help-section">
        <h2>训练指南</h2>
        <div class="training-info">
          <h3>训练类型</h3>
          <ul>
            <li><strong>单独训练</strong> - 针对单个模型进行专门训练</li>
            <li><strong>联合训练</strong> - 多个模型协同训练</li>
            <li><strong>持续训练</strong> - 系统持续学习和优化</li>
          </ul>
        </div>
      </section>

      <!-- Troubleshooting -->
      <section class="help-section">
        <h2>常见问题</h2>
        <div class="faq-list">
          <div class="faq-item">
            <h3>系统启动失败</h3>
            <p>检查系统日志和依赖项安装</p>
          </div>
          <div class="faq-item">
            <h3>模型无响应</h3>
            <p>检查模型状态和网络连接</p>
          </div>
          <div class="faq-item">
            <h3>训练失败</h3>
            <p>检查数据集格式和模型依赖</p>
          </div>
        </div>
      </section>
    </main>

    <!-- Footer -->
    <footer class="help-footer">
      <p>Self Soul System v2.0.0 | Apache 2.0 License</p>
    </footer>
  </div>
</template>

<script>
export default {
  name: 'HelpView',
  data() {
    return {
      // Minimal data for the component
    }
  },
  mounted() {
    document.title = 'Self Soul AGI 系统帮助'
  }
}
</script>

<style scoped>
/* Color Variables */
:root {
  --bg-primary: #ffffff;
  --bg-secondary: #f8f8f8;
  --bg-tertiary: #f0f0f0;
  --text-primary: #333333;
  --text-secondary: #666666;
  --text-tertiary: #999999;
  --border-color: #e0e0e0;
  --border-dark: #cccccc;
  --accent-color: #2c2c2c;
}

/* Container Styles */
.help-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
  min-height: 100vh;
  background-color: var(--bg-primary);
  color: var(--text-primary);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  line-height: 1.6;
}

/* Header Styles */
.help-header {
  text-align: center;
  margin-bottom: 3rem;
  padding-bottom: 1.5rem;
  border-bottom: 2px solid var(--border-color);
}

.help-header h1 {
  font-size: 2.5rem;
  font-weight: 300;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.help-header p {
  font-size: 1.1rem;
  color: var(--text-secondary);
  margin: 0;
}

/* Main Content Styles */
.help-main {
  margin-bottom: 3rem;
}

.help-section {
  margin-bottom: 2.5rem;
  padding: 2rem;
  background-color: var(--bg-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.help-section h2 {
  font-size: 1.8rem;
  font-weight: 400;
  margin-bottom: 1.5rem;
  color: var(--accent-color);
  border-bottom: 1px solid var(--border-color);
  padding-bottom: 0.5rem;
}

.help-section p {
  color: var(--text-secondary);
  margin-bottom: 1rem;
}

/* Step List Styles */
.step-list {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
}

.step-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: var(--accent-color);
  color: var(--bg-primary);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  flex-shrink: 0;
}

.step-content h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--text-primary);
}

.step-content p {
  margin: 0;
  color: var(--text-secondary);
}

/* Model Grid Styles */
.model-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin-top: 1rem;
}

.model-card {
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  transition: all 0.2s ease;
}

.model-card:hover {
  border-color: var(--border-dark);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.model-card h3 {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.model-card p {
  margin: 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

/* Training Info Styles */
.training-info h3 {
  font-size: 1.2rem;
  font-weight: 500;
  margin-bottom: 1rem;
  color: var(--text-primary);
}

.training-info ul {
  list-style: none;
  padding: 0;
}

.training-info li {
  padding: 0.5rem 0;
  color: var(--text-secondary);
  border-bottom: 1px solid var(--border-color);
}

.training-info li:last-child {
  border-bottom: none;
}

.training-info strong {
  color: var(--text-primary);
}

/* FAQ Styles */
.faq-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.faq-item {
  padding: 1.5rem;
  background-color: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
}

.faq-item h3 {
  font-size: 1.1rem;
  font-weight: 500;
  margin-bottom: 0.5rem;
  color: var(--accent-color);
}

.faq-item p {
  margin: 0;
  color: var(--text-secondary);
}

/* Footer Styles */
.help-footer {
  text-align: center;
  padding: 2rem 0;
  margin-top: 3rem;
  border-top: 1px solid var(--border-color);
  color: var(--text-tertiary);
  font-size: 0.9rem;
}

.help-footer p {
  margin: 0;
}

/* Responsive Design */
@media (max-width: 768px) {
  .help-container {
    padding: 1rem;
  }
  
  .help-header h1 {
    font-size: 2rem;
  }
  
  .help-section {
    padding: 1.5rem;
  }
  
  .model-grid {
    grid-template-columns: 1fr;
  }
  
  .step {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .step-number {
    margin-bottom: 0.5rem;
  }
}
</style>
