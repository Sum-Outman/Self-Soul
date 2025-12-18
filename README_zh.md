# Self Soul - 高级AGI系统

**开发团队**: Self Soul Team  
**开发人员邮箱**: silencecrowtom@qq.com  
**仓库地址**: https://github.com/Sum-Outman/Self-Soul

## 项目概述

Self Soul是一个复杂的通用人工智能(AGI)平台，将19个专业AI模型集成到统一的认知架构中。这个开源系统提供了全面的多模态智能能力，包括自然语言处理、计算机视觉、音频分析、情感智能、自主学习和高级推理。系统设计支持真正的从零开始训练、多摄像头视觉能力、外部设备集成，以及本地和外部API模型之间的无缝切换。

### 设计理念

Self Soul基于一个核心原则构建：真正的AGI需要一个凝聚、集成的架构，而不是孤立的模型。我们的设计理念强调：

- **统一认知架构**: 所有19个模型通过中央协调系统协同工作，实现超越单个组件总和的涌现智能
- **从零开始训练**: 通过从头开始训练所有模型而不依赖预训练基础，我们保持对模型开发、伦理对齐和AGI合规性的完全控制
- **以人为本的智能**: 集感情感智能和价值对齐，确保AI行为负责任、符合伦理并与人类价值观一致
- **模块化扩展性**: 灵活的架构允许轻松集成新的AI能力，同时保持系统一致性

### 技术亮点

- **19个专业AI模型**: 全面覆盖从基本感知到高级推理的认知能力
- **先进的多模态集成**: 无缝处理和融合文本、图像、音频和视频数据
- **自适应学习引擎**: 基于性能指标实时优化学习策略和训练参数
- **分布式处理**: 每个模型在专用端口(8001-8019)上运行，实现并行处理和可扩展性
- **现代UI/UX**: 基于Vue.js的直观仪表板，用于系统管理和监控
- **全面的API**: RESTful接口，用于与外部系统和应用程序集成
- **多模态数据集支持**: 扩展的Multimodal Dataset v1支持所有19个模型的全面训练

Self Soul代表了AGI研究和开发的重大进步，为探索通用人工智能前沿提供了功能齐全的平台。

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Sum-Outman/Self-Soul)](https://github.com/Sum-Outman/Self-Soul/issues)

### 核心功能

#### 多模态集成

无缝结合文本、图像、音频和视频处理，采用统一的认知架构。这使系统能够同时理解和处理多种格式的信息，对环境形成更全面的理解。

**使用场景**: 媒体分析、内容创作、多感官用户界面、环境感知系统

#### 自主学习

具有持续适应能力的高级自主学习系统。系统可以在没有明确人类指令的情况下独立提高其性能。

- 基于实时性能指标的**自动模型改进**
- 识别并关注模型薄弱环节的**自适应课程学习**
- 使用元强化学习的**实时学习优化**
- 跨所有19种AI能力的**多模型协调**同步自主学习
- 用于策略优化的**内置自适应学习引擎**

**使用场景**: 自我改进的客户服务系统、自适应教育平台、自主研究助手

#### 自主训练

通过http://localhost:5175/#/training的训练界面提供全面的自主训练能力。用户可以通过最少的配置启动完全自主的训练会话。

- 通过直观的Web界面访问**完全自主训练模式**
- 基于模型性能的**自优化训练参数**(学习率、批处理大小、轮次)
- 基于模型能力和训练目标的**智能数据集选择**
- 跨所有19种AI能力的**多模型协调**同步自主训练
- 用于参数调整的**内置自适应学习引擎**
- 最少人工监督的**持续训练循环**

#### 训练界面功能

训练界面提供了一套全面的工具来管理模型训练：

- **训练模式选择**: 在单独或联合训练模式之间选择
  - **单独模式**: 一次训练一个模型，使用聚焦参数
  - **联合模式**: 同时训练多个模型，进行协调优化

- **模型选择系统**:
  - **推荐组合**: 针对常见用例的预配置模型组合
  - **全选模型**: 快速选择所有19个AI模型进行全面训练
  - **模型网格**: 带有工具提示和状态指示器的可视化选择界面
  - **依赖关系可视化**: 显示联合训练验证的模型依赖关系

- **数据集管理**:
  - **多模态数据集v1**: 支持所有19个模型，具有全面的格式支持
  - **专用数据集**: 仅语言、仅视觉、仅传感器和其他特定领域的数据集
  - **上传功能**: 支持上传带有格式验证的自定义数据集
  - **支持的模型显示**: 显示每个数据集的兼容模型

- **参数配置**:
  - **基本参数**: 轮次、批处理大小、学习率、验证拆分
  - **高级参数**: 丢弃率、权重衰减、动量、优化器选择
  - **从零开始选项**: 在从零开始训练或使用现有权重之间切换

- **训练策略选项**:
  - **默认策略**: 适用于大多数用例的平衡方法
  - **知识辅助训练**: 利用现有知识库加快学习速度
  - **自适应课程学习**: 根据性能动态调整训练难度
  - **迁移学习**: 促进相关模型之间的知识转移

- **实时状态更新**:
  - **进度可视化**: 带有指标和图表的实时训练进度
  - **错误处理**: 清晰的错误消息和故障排除建议
  - **成功反馈**: 已完成操作的确认消息

- **AGI训练状态监控**:
  - **元学习进度**: 跟踪自我改进能力
  - **知识集成水平**: 衡量模型集成新知识的程度
  - **自主学习分数**: 量化自主学习效率
  - **自适应学习效率**: 评估参数优化效果

**使用场景**: 快速模型开发、持续系统改进、大规模训练操作

#### 高级语言处理

支持多种语言，具有深度上下文理解和推理能力。语言模型采用自定义的基于Transformer的架构，用于复杂的文本处理。

**使用场景**: 跨语言通信、内容生成、文档分析、对话AI

#### 实时监控

全面的系统性能跟踪和模型指标可视化。监控系统提供对模型行为和系统资源使用情况的实时洞察。

**使用场景**: 系统优化、性能调试、资源分配、训练进度跟踪

#### 交互式仪表板

基于Vue.js的现代界面，用于直观的系统管理。仪表板提供了一个集中位置，用于控制Self Soul系统的所有方面。

**使用场景**: 系统管理、模型配置、训练管理、性能监控

#### 可扩展架构

模块化设计，允许轻松集成新的AI能力。系统的基于组件的架构支持无缝添加新模型和功能。

**使用场景**: 自定义AI能力集成、第三方系统连接、专业应用开发

#### 知识管理

具有结构化和非结构化数据集成的高级知识库。知识模型以一种能够高效检索和推理的方式组织信息。

**使用场景**: 企业知识管理、研究辅助、信息检索系统、决策支持工具

#### 情感智能

用于类人交互的情感识别和响应能力。情感模型可以识别并适当地响应用户的情绪状态。

**使用场景**: 客户服务自动化、心理健康支持、个性化用户体验、社交机器人

#### 高级推理

复杂的逻辑推理和创造性问题解决能力。推理模型使系统能够解决复杂问题并生成创新解决方案。

**使用场景**: 科学研究辅助、工程设计、战略规划、创意内容生成

#### 值对齐

伦理指南集成，确保负责任的AI行为。值对齐模型确保系统的行动与人类价值观和伦理原则一致。

**使用场景**: 伦理AI应用、合规监控、负责任决策系统、安全AGI开发

## 系统架构

Self Soul采用分层架构，将核心AI能力与用户界面分离，支持模块化开发和部署。该架构设计支持19个专业AI模型的协调，同时提供统一的用户体验。

### 架构概述

Self Soul架构由三个主要层组成：

1. **表示层**: 基于Vue.js的前端应用程序，提供系统管理、监控和交互的用户界面
2. **API层**: RESTful API网关，处理前端和后端服务之间的通信
3. **核心层**: 分布式AI模型和管理服务，构成系统的核心

### 架构图

```
Self Soul /
├── core/                     # 核心后端系统
│   ├── main.py               # 后端入口点和API端点
│   ├── model_service_manager.py # 模型服务创建和管理
│   ├── model_registry.py     # 模型注册和生命周期管理
│   ├── training_manager.py   # 训练协调和管理
│   ├── autonomous_learning_manager.py # 自主学习能力
│   ├── joint_training_coordinator.py  # 多模型训练协调
│   └── error_handling.py     # 错误处理和日志系统
├── app/                      # 前端应用
│   ├── src/                  # 前端源代码
│   │   ├── views/            # 不同视图的Vue组件
│   │   ├── components/       # 可重用UI组件
│   │   └── assets/           # 静态资源
│   ├── public/               # 可公开访问的文件
│   └── package.json          # 前端依赖
├── config/                   # 系统配置文件
│   └── model_services_config.json # 模型服务端口配置
├── data/                     # 模型和训练的数据存储
├── requirements.txt          # Python依赖
└── README.md                 # 项目文档
```

### 核心组件

#### 前端组件

- **主仪表板**: 系统状态和快速操作的中央枢纽
- **模型管理**: 配置和控制单个AI模型的界面
- **训练界面**: 管理模型训练过程的全面工具
- **知识库**: 管理系统知识存储库的界面
- **设置面板**: 系统行为和偏好的配置选项
- **帮助中心**: 文档和故障排除资源

#### 后端服务

- **主API网关(端口8000)**: 所有前端请求的中央入口点
- **模型服务管理器**: 创建和管理所有19个AI模型的实例
- **模型注册表**: 维护有关所有可用模型及其能力的信息
- **训练管理器**: 协调所有模型的训练活动
- **自主学习管理器**: 监督自主学习过程和优化
- **联合训练协调器**: 管理多模型训练同步
- **错误处理系统**: 集中式日志和错误管理

### 组件交互流程

Self Soul组件通过一组定义明确的通信路径进行交互：

#### 用户请求流程

1. **用户操作**: 用户与前端界面交互(例如，启动模型训练)
2. **前端处理**: Vue组件处理请求并准备API调用
3. **API请求**: 前端将请求发送到主API网关(端口8000)
4. **API路由**: 网关将请求路由到适当的后端服务
5. **服务处理**: 目标服务处理请求(例如，训练管理器)
6. **模型交互**: 服务与相关AI模型交互(例如，在特定模型上启动训练)
7. **响应流程**: 结果通过相同的路径传回用户界面

#### 模型协调流程

1. **管理器模型请求**: 管理器模型(端口8001)识别对专业模型的需求
2. **服务发现**: 模型服务管理器定位适当的模型服务
3. **模型调用**: 通过其专用端口(8002-8019)调用相关模型
4. **结果处理**: 模型处理请求并返回结果
5. **协调响应**: 管理器模型集成结果并返回最终响应

#### 训练流程

1. **训练启动**: 用户或自主学习管理器请求模型训练
2. **数据集选择**: 训练管理器基于模型能力选择适当的数据集
3. **训练配置**: 参数由自适应学习引擎优化
4. **模型训练**: 各个模型在其专用端口上训练
5. **进度监控**: 实时收集和显示指标
6. **模型更新**: 训练好的模型被保存并重新加载到系统中

### 数据流

- **输入数据**: 用户输入、上传的文件和外部数据源
- **处理管道**: 数据根据类型和上下文流经适当的模型
- **知识集成**: 处理后的信息被集成到知识库中
- **输出生成**: 最终结果被格式化为用户呈现或系统使用

Self Soul的模块化架构确保组件可以独立开发、测试和部署，同时与系统的其余部分保持无缝集成。

## 从零开始训练架构

**Self Soul为所有19个模型实现了完整的从零开始训练架构，不依赖任何预训练模型。**这确保了对模型开发、伦理对齐和AGI合规性的完全控制。

### 核心训练原则

Self Soul的训练架构遵循四个基本原则：

- **无预训练模型**: 所有神经网络都是使用PyTorch从头开始构建和训练的，确保对每个参数和层的完全控制
- **自定义架构**: 每种模型类型都有专门为其特定认知功能量身定制的神经网络设计
- **AGI兼容设计**: 模型遵循AGI原则的统一认知架构，通过模型协调实现涌现智能
- **自主改进**: 内置的自主学习和元认知能力允许模型在没有人类干预的情况下不断改进

### 从零开始实现细节

#### 训练管道架构

从零开始的训练管道由几个关键组件组成：

1. **数据准备层**: 原始数据处理、标准化和增强
2. **模型架构层**: 每个专业模型的自定义神经网络设计
3. **训练执行层**: 在专用端口(8001-8019)上的分布式训练
4. **优化层**: 用于实时参数调整的自适应学习引擎
5. **评估层**: 全面的性能指标和验证
6. **持久化层**: 模型保存、加载和版本管理

#### 神经网络设计方法

每个模型都采用专门为其认知功能设计的专用神经网络架构：

- **管理器模型**: 具有注意力机制的分层协调网络，用于任务分配
- **语言模型**: 具有可变大小注意力窗口和动态层配置的自定义基于Transformer的架构
- **视觉模型**: 具有自适应池化和多尺度特征提取的混合CNN架构
- **音频模型**: 用于时间音频信号处理的循环卷积网络
- **推理模型**: 具有符号推理能力的图神经网络
- **情感模型**: 具有注意力机制的密集神经网络，用于情感模式识别

#### 训练数据管理

- **多模态数据集v1**: 支持所有19个模型的综合数据集，包含：
  - 文本: 纯文本、结构化文档、代码存储库
  - 图像: 跨多个领域的照片、图表、图表
  - 音频: 带有元数据的语音、音乐、环境声音
  - 视频: 带有同步音频的多摄像头录像
- **数据增强**: 为每种数据类型量身定制的动态增强策略
- **数据集兼容性**: 前端`supportedModels`属性确保数据集-模型对齐
- **后端验证**: `dataset_manager.py`根据模型要求验证文件格式

### 模型架构概述

#### 基础模型(从零开始实现)

- **管理器模型(端口8001)**: CoordinationNeuralNetwork, TaskAllocationNetwork - 使用分层注意力机制管理系统资源并协调其他模型
- **语言模型(端口8002)**: LanguageNeuralNetwork, FromScratchLanguageTrainer - 具有12层、8个注意力头和768个隐藏维度的自定义Transformer架构
- **知识模型(端口8003)**: KnowledgeGraphNetwork, SemanticEmbeddingLayer - 用于结构化和非结构化知识组织的基于图的架构
- **视觉模型(端口8004)**: SimpleVisionCNN, VisionDataset - 具有5个卷积层、自适应池化和多尺度特征融合的自定义CNN
- **音频模型(端口8005)**: AudioRecognitionNetwork, SpectrogramLayer - 用于声音和语音理解的循环卷积架构
- **自主模型(端口8006)**: AdvancedDecisionNetwork, ExperienceReplayBuffer - 具有经验回放和价值函数的强化学习架构

#### 高级模型(完全从零开始训练)

- **编程模型(端口8007)**: ProgrammingNeuralNetwork, CodeEmbeddingLayer - 用于代码生成和优化的语法感知神经网络
- **规划模型(端口8008)**: PlanningStrategyNetwork, StepPredictionNetwork - 具有时间推理的分层规划架构
- **情感模型(端口8009)**: EmotionRecognitionNetwork, AffectiveLayer - 用于情感智能的具有注意力机制的密集网络
- **空间模型(端口8010)**: SpatialNeuralNetwork, GeometricEmbeddingLayer - 具有坐标变换的3D空间推理架构
- **计算机视觉模型(端口8011)**: VisualUnderstandingNetwork, ObjectDetectionLayer - 用于对象检测和场景理解的高级CNN
- **传感器模型(端口8012)**: SensorNeuralNetwork, SignalProcessingLayer - 用于多模态传感器数据融合的架构
- **运动模型(端口8013)**: TrajectoryPlanningNetwork, KinematicsLayer - 物理感知运动预测和规划
- **预测模型(端口8014)**: PredictionNeuralNetwork (LSTM+Attention), TimeSeriesLayer - 具有注意力机制的序列预测架构
- **高级推理模型(端口8015)**: ReasoningGraphNetwork, LogicLayer - 具有符号集成的基于图的推理架构
- **数据融合模型(端口8016)**: FusionNeuralNetwork, CrossModalAttentionLayer - 具有跨模态注意力的多源信息集成
- **创造性问题解决模型(端口8017)**: CreativeGenerationNetwork, AnalogicalReasoningLayer - 具有组合创新能力的神经生成器
- **元认知模型(端口8018)**: MetaLearningNetwork, SelfReflectionLayer - 基于经验的学习和自我监控架构
- **值对齐模型(端口8019)**: EthicalReasoningNetwork, ValueSystemLayer - 自定义标记化和值对齐架构

### 训练系统功能

#### 统一训练接口

所有模型都实现了标准化的训练接口：
- `enable_training()`: 初始化训练模式并准备资源
- `disable_training()`: 完成训练并保存模型状态
- `train_step()`: 执行单个训练迭代
- `evaluate()`: 运行验证并返回性能指标

#### 自主训练接口

http://localhost:5175/#/training的高级训练接口提供：
- 最少用户配置的**一键自主训练模式**
- 基于实时模型性能指标的**自动参数优化**
- 随模型能力发展的**自适应训练调度**和动态课程设计
- 带有可行自主改进建议的**实时进度可视化**
- 基于训练目标和模型能力的**智能数据集选择**
- 跨所有19种AI能力的**多模型协调**同步自主学习
- 与支持所有模型进行全面训练的多模态数据集v1的**无缝集成**
- 当达到最佳结果时自动终止训练的**自动性能监控**

#### 自适应学习引擎

Self Soul自主训练能力的核心是自适应学习引擎：
- **实时参数调整**: 动态调整学习率、批处理大小和正则化
- **基于性能的优化**: 使用验证指标优化模型超参数
- **课程学习**: 根据模型性能动态调整训练难度
- **模型间迁移学习**: 支持专业模型之间的知识转移
- **早期停止检测**: 当性能达到平台期时自动终止训练

#### 模型持久化和版本控制

- **标准化保存/加载**: 所有模型都实现`save_model()`和`load_model()`方法
- **检查点系统**: 在训练过程中定期保存模型状态
- **版本管理**: 跟踪模型版本和训练参数
- **兼容层**: 确保模型版本之间的向后兼容性

#### 分布式训练架构

- **专用端口**: 每个模型在专用端口(8001-8019)上运行，用于并行处理
- **集中管理**: `FromScratchTrainingManager`协调所有训练活动
- **同步更新**: 确保整个系统的模型改进一致
- **资源优化**: 根据训练需求动态分配系统资源

#### 多模态数据集v1

扩展的多模态数据集v1为所有模型提供全面支持：
- **格式支持**: 每种模型类型的多种文件格式(例如，JSON、CSV、TXT用于语言；JPG、PNG用于视觉)
- **数据集-模型映射**: 前端配置中明确的`supportedModels`属性
- **后端验证**: `dataset_manager.py`根据模型要求验证文件格式
- **可扩展设计**: 以最少的配置更改支持未来的模型添加

### 技术挑战和解决方案

#### 挑战: 同时训练19个模型
**解决方案**: 具有专用端口和集中协调的分布式架构

#### 挑战: 维护模型一致性
**解决方案**: 统一的认知架构和模型之间的知识转移

#### 挑战: 跨不同模型类型的优化
**解决方案**: 具有模型特定优化策略的自适应学习引擎

#### 挑战: 确保伦理对齐
**解决方案**: 集成到所有决策过程中的值对齐模型(端口8019)

Self Soul的从零开始训练架构代表了AGI开发的重大突破，使系统能够在全面的认知能力集上实现真正的自主学习和适应。

## 安装指南

### 系统要求

- Windows 10/11, Linux或macOS
- Python 3.9+(推荐: 3.11)
- Node.js 16+(推荐: 18)
- 8GB RAM(最小), 推荐16GB+
- 10GB可用磁盘空间

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Sum-Outman/Self-Soul
   cd Self-Soul
   ```

2. **设置Python虚拟环境**
   ```bash
   python -m venv .venv
   ```

3. **激活虚拟环境**
   - Windows:
   ```bash
   .venv\Scripts\activate
   ```
   - Linux/macOS:
   ```bash
   source .venv/bin/activate
   ```

4. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **安装前端依赖**
   ```bash
   cd app
   npm install
   ```

6. **配置环境变量**
   创建一个`.env`文件并添加以下内容：
   ```
   API_KEY=your-api-key-here
   MODEL_PATH=./models
   DATA_PATH=./data
   ```

7. **启动系统**
   ```bash
   # 启动后端服务
   python core/main.py
   
   # 在新终端中，启动前端应用
   cd app && npm run dev
   ```

8. **访问系统**
   打开浏览器并访问: http://localhost:5175

## 使用指南

### 主要功能

1. **仪表板**: 监控系统性能和活动模型
2. **模型管理**: 配置和控制单个AI模型
3. **知识管理**: 导入、浏览和管理知识库
4. **训练**: 使用自定义数据集训练和微调模型
5. **设置**: 配置系统偏好和语言设置
6. **帮助中心**: 访问文档和使用指南

### 语言支持

- 英语

### 详细配置

#### 环境变量

在项目根目录中创建一个`.env`文件，包含以下配置：

```
# API配置
API_KEY=your-api-key-here
API_BASE_URL=http://localhost:8000

# 模型配置
MODEL_PATH=./models
DATA_PATH=./data
TRAINING_PATH=./training

# 系统配置
LOG_LEVEL=INFO
PORT=8000
WS_PORT=8001

# 数据库配置
DATABASE_URL=sqlite:///./self_soul.db

# 外部服务配置
EXTERNAL_API_URL=https://api.external-service.com
EXTERNAL_API_KEY=your-external-api-key-here
```

#### 模型配置

模型配置存储在`config/model_services_config.json`中：

```json
{
  "model_services": [
    {"id": "manager", "name": "Manager Model", "port": 8001, "enabled": true},
    {"id": "language", "name": "Language Model", "port": 8002, "enabled": true},
    {"id": "knowledge", "name": "Knowledge Model", "port": 8003, "enabled": true},
    {"id": "vision", "name": "Vision Model", "port": 8004, "enabled": true},
    {"id": "audio", "name": "Audio Model", "port": 8005, "enabled": true},
    {"id": "autonomous", "name": "Autonomous Model", "port": 8006, "enabled": true},
    {"id": "programming", "name": "Programming Model", "port": 8007, "enabled": true},
    {"id": "planning", "name": "Planning Model", "port": 8008, "enabled": true},
    {"id": "emotion", "name": "Emotion Model", "port": 8009, "enabled": true},
    {"id": "spatial", "name": "Spatial Model", "port": 8010, "enabled": true},
    {"id": "computer_vision", "name": "Computer Vision Model", "port": 8011, "enabled": true},
    {"id": "sensor", "name": "Sensor Model", "port": 8012, "enabled": true},
    {"id": "motion", "name": "Motion Model", "port": 8013, "enabled": true},
    {"id": "prediction", "name": "Prediction Model", "port": 8014, "enabled": true},
    {"id": "advanced_reasoning", "name": "Advanced Reasoning Model", "port": 8015, "enabled": true},
    {"id": "data_fusion", "name": "Data Fusion Model", "port": 8016, "enabled": true},
    {"id": "creative_problem_solving", "name": "Creative Problem Solving Model", "port": 8017, "enabled": true},
    {"id": "meta_cognition", "name": "Meta Cognition Model", "port": 8018, "enabled": true},
    {"id": "value_alignment", "name": "Value Alignment Model", "port": 8019, "enabled": true}
  ]
}
```

### 使用示例

#### 1. 系统初始化

启动后端服务：

```bash
python core/main.py
```

启动前端应用：

```bash
cd app && npm run dev
```

#### 2. 模型管理

**检查模型状态**:
```bash
curl -X GET http://localhost:8000/api/models/status
```

**切换模型到外部API**:
```bash
curl -X POST http://localhost:8000/api/models/language/external \
  -H "Content-Type: application/json" \
  -d '{"api_url": "https://api.external-service.com/language", "api_key": "your-api-key"}'
```

**切换模型到本地**:
```bash
curl -X POST http://localhost:8000/api/models/language/local
```

#### 3. 训练

**通过API启动训练**:
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "language", "dataset_id": "multimodal_v1", "parameters": {"epochs": 10, "batch_size": 32, "learning_rate": 0.001}}'
```

**监控训练进度**:
```bash
# 连接到WebSocket端点获取实时更新
wscat -c ws://localhost:8000/ws/training/12345
```

#### 4. 知识管理

**导入知识**:
```bash
curl -X POST http://localhost:8000/api/knowledge/import \
  -H "Content-Type: application/json" \
  -d '{"title": "Example Knowledge", "content": "This is an example of imported knowledge."}'
```

**搜索知识**:
```bash
curl -X GET "http://localhost:8000/api/knowledge/search?q=example"
```

#### 5. 聊天界面

**与语言模型聊天**:
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "session_id": "test_session", "model_id": "language"}'
```

**与管理器模型聊天**:
```bash
curl -X POST http://localhost:8000/api/models/8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Coordinate with all models to analyze this text", "session_id": "test_session", "text": "This is a test text for analysis."}'
```

### 高级用法

#### 批量模型操作

**批量切换模型模式**:
```bash
curl -X POST http://localhost:8000/api/models/batch/switch \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision", "audio"], "mode": "external", "api_config": {"api_url": "https://api.external-service.com"}}'
```

**批量更新模型参数**:
```bash
curl -X POST http://localhost:8000/api/models/batch/update \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision"], "parameters": {"max_tokens": 1000, "temperature": 0.7}}'
```

#### 自主学习

**开始自主学习**:
```bash
curl -X POST http://localhost:8000/api/autonomous-learning/start \
  -H "Content-Type: application/json" \
  -d '{"model_ids": ["language", "vision"], "duration": 3600, "objective": "improve_comprehension"}'
```

**停止自主学习**:
```bash
curl -X POST http://localhost:8000/api/autonomous-learning/stop
```

### 页面功能指南

#### 首页(http://localhost:5175/#/)

首页作为系统监控和控制的主仪表板，具有：

- **实时系统监控**: 所有19个AI模型的实时状态更新，带有视觉指示器
- **设备管理**: 多摄像头控制、传感器数据可视化和串行通信接口
- **WebSocket通信**: 实时设备控制和状态更新
- **立体视觉处理**: 具有相机校准和深度感知的高级视觉能力
- **系统健康仪表板**: 性能指标、资源使用情况和错误警报
- **快速访问按钮**: 一键导航到其他主页面

#### 对话页面(http://localhost:5175/#/conversation)

对话页面启用与AGI系统的多模态交互：

- **多模态消息**: 支持文本、图像、音频和视频输入
- **模型连接状态**: 管理模型(端口8001)连接的实时指示
- **情感分析**: 情感检测结果和置信度分数的显示
- **消息历史**: 带有时间戳的持久对话日志
- **错误处理**: 当模型连接丢失时优雅降级
- **响应式设计**: 适应不同屏幕尺寸的自适应布局

#### 训练页面(http://localhost:5175/#/training)

训练页面提供了用于模型训练和优化的全面工具：

- **训练模式**: 单独(单个模型)或联合(多个模型)训练选项
- **模型选择**: 支持选择单个模型或推荐组合
- **依赖管理**: 用于联合训练的模型依赖关系可视化
- **数据集选择**: 支持多模态数据集v1和专用数据集
- **训练参数**: 可配置的轮次、批处理大小、学习率和正则化
- **训练策略**: 知识辅助训练和预训练微调选项
- **进度跟踪**: 通过WebSocket进行实时监控，带有回退轮询
- **评估指标**: 准确性、损失和混淆矩阵可视化
- **训练历史**: 以前训练会话的日志和结果

#### 知识页面(http://localhost:5175/#/knowledge)

知识页面使用高级文件处理管理系统的知识库：

- **导入功能**: 支持上传PDF、DOCX、TXT、JSON和CSV文件
- **领域分类**: 知识文件的自动和手动分类
- **文件管理**: 预览、下载、删除和组织知识条目
- **自动学习**: 带有进度跟踪的定时知识处理
- **统计仪表板**: 跨领域知识分布的可视化表示
- **WebSocket集成**: 自动学习进度的实时更新

#### 设置页面(http://localhost:5175/#/settings)

设置页面允许全面配置系统参数：

- **模型配置**: 在本地和外部API模型之间切换
- **API设置**: 配置API密钥、URL和连接参数
- **硬件设置**: 相机分辨率、串行端口波特率和传感器配置
- **批量操作**: 同时启动、停止或重启所有模型
- **系统管理**: 重启系统服务和重置配置
- **连接测试**: 验证与外部API模型的连接

### 界面使用示例

#### 首页导航

```bash
# 访问首页
echo "在浏览器中打开 http://localhost:5175/#/"
```

#### 训练工作流程

1. **访问训练页面**: 导航到`http://localhost:5175/#/training`
2. **选择训练模式**: 选择"单独"或"联合"模式
3. **选择模型**: 从网格中选择模型或使用推荐组合
4. **配置参数**: 设置轮次、批处理大小和学习率
5. **开始训练**: 点击"开始训练"并监控进度

#### 知识管理

1. **访问知识页面**: 导航到`http://localhost:5175/#/knowledge`
2. **上传文件**: 点击"导入知识"并选择要上传的文件
3. **分类内容**: 为上传的文件分配领域或使用自动分类
4. **查看统计**: 检查跨领域的知识分布
5. **开始自动学习**: 启动知识处理以改进模型

## 端口配置

Self Soul系统使用以下端口配置来运行各种服务和模型：

### 主要服务端口

| 主API网关 | Main API Gateway | 8000 | 系统的主要入口点，提供RESTful API接口 |
|---------|-----------------|------|----------------------------------|
| 前端应用 | Frontend Application | 5175 | 用户界面，可通过浏览器访问 |
| 实时数据流管理器 | Realtime Stream Manager | 8766 | 管理实时数据流和模型间通信 |
| 性能监控服务 | Performance Monitoring | 8080 | 监控系统性能和资源使用情况 |

### 模型端口配置

系统为每个AI模型分配了独立的端口，范围从8001到8019：

| 端口号 | 模型类型 | 英文模型类型 |
|-------|---------|------------|
| 8001 | 管理模型 | Manager Model |
| 8002 | 语言模型 | Language Model |
| 8003 | 知识模型 | Knowledge Model |
| 8004 | 视觉模型 | Vision Model |
| 8005 | 音频模型 | Audio Model |
| 8006 | 自主模型 | Autonomous Model |
| 8007 | 编程模型 | Programming Model |
| 8008 | 规划模型 | Planning Model |
| 8009 | 情感模型 | Emotion Model |
| 8010 | 空间模型 | Spatial Model |
| 8011 | 计算机视觉模型 | Computer Vision Model |
| 8012 | 传感器模型 | Sensor Model |
| 8013 | 运动模型 | Motion Model |
| 8014 | 预测模型 | Prediction Model |
| 8015 | 高级推理模型 | Advanced Reasoning Model |
| 8016 | 数据融合模型 | Data Fusion Model |
| 8017 | 创造性问题解决模型 | Creative Problem Solving Model |
| 8018 | 元认知模型 | Meta Cognition Model |
| 8019 | 值对齐模型 | Value Alignment Model |

端口配置存储在`config/model_services_config.json`文件中，系统启动时会自动加载这些配置。

## API文档

Self Soul提供了全面的RESTful API和WebSocket端点，用于与其他应用程序集成。API按类别组织，便于导航和使用。

### 基础URL

```
http://localhost:8000
```

### API类别

#### 1. 系统状态和健康

| 端点 | 方法 | 描述 | 响应 |
|----------|--------|-------------|----------|
| `/health` | GET | 系统健康检查 | `{"status": "ok", "message": "Self Soul system is running normally"}` |
| `/api/models/status` | GET | 获取所有模型的状态 | `{"status": "success", "data": {model_id: {status_info}}}` |
| `/api/models/language/status` | GET | 获取语言模型的状态 | `{"status": "success", "data": {language_model_status}}` |
| `/api/models/management/status` | GET | 获取管理模型的状态 | `{"status": "success", "data": {management_model_status}}` |
| `/api/models/from_scratch/status` | GET | 获取从零开始模型的状态 | `{"status": "success", "data": {from_scratch_status}}` |

#### 2. 模型管理

| 端点 | 方法 | 描述 | 响应 |
|----------|--------|-------------|----------|
| `/api/models/getAll` | GET | 获取所有模型的信息 | `{"status": "success", "data": [models]}` |
| `/api/models/available` | GET | 获取前端选择可用的模型 | `{"status": "success", "models": [available_models]}` |
| `/api/models/config` | GET | 获取所有模型的配置 | `{"status": "success", "data": {model_configs}}` |

#### 3. 数据处理

| 端点 | 方法 | 描述 | 请求体 | 响应 |
|----------|--------|-------------|--------------|----------|
| `/api/process/text` | POST | 处理文本输入 | `{"text": "text_to_process", "lang": "en"}` | `{"status": "success", "data": {processing_result}}` |
| `/api/chat` | POST | 与语言模型聊天 | `{"message": "user_message", "session_id": "session_123", "conversation_history": []}` | `{"status": "success", "data": {response, conversation_history}}` |
| `/api/models/8001/chat` | POST | 与管理器模型聊天 | `{"message": "user_message", "session_id": "session_123", "conversation_history": []}` | `{"status": "success", "data": {response, conversation_history}}` |

#### 4. 设备和外部接口

| 端点 | 方法 | 描述 | 响应 |
|----------|--------|-------------|----------|
| `/api/devices/cameras` | GET | 获取可用相机列表 | `{"status": "success", "data": [cameras]}` |
| `/api/serial/ports` | GET | 获取可用串行端口列表 | `{"status": "success", "data": [serial_ports]}` |
| `/api/devices/external` | GET | 获取所有外部设备的信息 | `{"status": "success", "data": [devices]}` |
| `/api/serial/connect` | POST | 连接到串行端口设备 | `{"device_id": "dev_123", "port": "COM3", "baudrate": 9600}` | `{"status": "success", "data": {connection_result}}` |
| `/api/serial/disconnect` | POST | 断开与串行端口设备的连接 | `{"device_id": "dev_123"}` | `{"status": "success", "data": {disconnection_result}}` |

#### 5. WebSocket端点

| 端点 | 描述 |
|----------|-------------|
| `/ws/training/{job_id}` | 实时训练进度更新 |
| `/ws/monitoring` | 系统监控数据 |
| `/ws/test-connection` | WebSocket连接测试 |
| `/ws/autonomous-learning/status` | 自主学习状态更新 |
| `/ws/audio-stream` | 实时音频流处理 |
| `/ws/video-stream` | 实时视频流处理 |

### API使用示例

#### 健康检查

```bash
curl -X GET http://localhost:8000/health
```

响应:
```json
{"status": "ok", "message": "Self Soul system is running normally"}
```

#### 与语言模型聊天

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?", "session_id": "test_session"}'
```

响应:
```json
{
  "status": "success",
  "data": {
    "response": "I'm doing well, thank you for asking!",
    "conversation_history": [
      {"role": "user", "content": "Hello, how are you?"},
      {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ],
    "session_id": "test_session"
  }
}
```

#### 获取模型状态

```bash
curl -X GET http://localhost:8000/api/models/status
```

响应:
```json
{
  "status": "success",
  "data": {
    "manager": {"id": "manager", "name": "Manager Model", "status": "active", ...},
    "language": {"id": "language", "name": "Language Model", "status": "active", ...},
    ...
  }
}
```

### API文档访问

要使用Swagger UI访问交互式API文档，请访问：

http://localhost:8000/docs (当后端运行时)

这个交互式文档提供了所有API端点、请求参数和响应格式的详细信息。您还可以直接从Swagger UI界面测试API调用。

## 常见问题和解决方案

### 虚拟环境问题

如果您遇到虚拟环境问题：
```bash
python -c "import shutil; shutil.rmtree('.venv', ignore_errors=True)"
python -m venv .venv
```

### 依赖安装问题

如果pip安装失败：
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 前端启动问题

如果npm run dev失败：
```bash
cd app
rm -rf node_modules
npm install
npm run dev
```

## 贡献

我们欢迎对Self Soul项目的贡献。请按照以下步骤操作：

1. Fork仓库
2. 创建功能分支
3. 提交您的更改
4. 推送到分支
5. 创建拉取请求

请确保您的代码符合项目的编码标准，并包含适当的测试。

## 许可证

本项目采用Apache License 2.0许可证。有关详细信息，请参阅[LICENSE](LICENSE)文件。

## 致谢

Self Soul是在开源社区的贡献下构建的，并利用了各种AI研究和技术。

---

© 2025 Self Soul Team. 保留所有权利。