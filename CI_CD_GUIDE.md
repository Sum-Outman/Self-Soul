# Self-Soul-B多模态AGI系统CI/CD指南

本文档描述了Self-Soul-B系统的持续集成和持续部署（CI/CD）流水线配置、使用方法和最佳实践。

## 📋 目录
- [CI/CD概述](#cicd概述)
- [工作流配置](#工作流配置)
- [本地开发集成](#本地开发集成)
- [环境管理](#环境管理)
- [安全最佳实践](#安全最佳实践)
- [故障排除](#故障排除)
- [扩展和自定义](#扩展和自定义)

## 🎯 CI/CD概述

Self-Soul-B采用现代化的CI/CD流水线，确保代码质量、自动化测试和可靠部署。

### CI/CD目标
1. **代码质量保证**: 自动化代码检查和格式化
2. **自动化测试**: 运行单元测试、集成测试和性能测试
3. **持续集成**: 频繁集成代码变更，快速发现问题
4. **持续部署**: 自动化部署到测试和生产环境
5. **发布管理**: 标准化版本发布流程

### 流水线架构
```
┌─────────────────────────────────────────────────────────┐
│                   开发者本地开发                         │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                GitHub Actions CI流水线                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │ 代码质量 │ │ 单元测试 │ │ 集成测试 │ │ 安全扫描 │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                CD部署流水线                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │
│  │构建镜像  │ │部署测试  │ │部署生产  │ │发布版本  │   │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │
└─────────────────────────────────────────────────────────┘
```

## ⚙️ 工作流配置

### 1. 持续集成工作流 (`.github/workflows/ci.yml`)

#### 触发条件
- 推送代码到 `main` 或 `develop` 分支
- 创建拉取请求到 `main` 或 `develop` 分支
- 忽略对文档文件的更改（`.md`, `.txt`）

#### 工作流任务
| 任务名称 | 描述 | 依赖 |
|----------|------|------|
| **代码质量检查** | 检查代码格式、风格、类型 | - |
| **后端测试** | 运行Python单元测试和集成测试 | 代码质量检查 |
| **前端测试** | 运行前端测试（如果存在） | 代码质量检查 |
| **性能测试** | 运行性能基准测试 | 后端测试 |
| **Docker构建测试** | 测试Docker镜像构建 | 后端测试 |
| **安全扫描** | 运行安全漏洞扫描 | - |
| **发布工件** | 创建可部署的工件 | 后端测试、Docker构建、安全扫描 |

#### 配置示例
```yaml
# 手动触发CI流水线
on:
  workflow_dispatch:
  
# 或通过API触发
curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/OWNER/REPO/actions/workflows/ci.yml/dispatches \
  -d '{"ref":"main"}'
```

### 2. 持续部署工作流 (`.github/workflows/deploy.yml`)

#### 触发条件
- 手动触发（通过GitHub Actions界面）
- 推送代码到 `main` 分支（核心文件更改）

#### 部署环境
| 环境 | 描述 | 触发条件 |
|------|------|----------|
| **测试环境** | 预生产环境，用于验证 | 推送到main分支 |
| **生产环境** | 生产环境，服务真实用户 | 手动触发 |

#### 部署策略
- **蓝绿部署**: 最小化停机时间，快速回滚
- **健康检查**: 部署前验证服务状态
- **自动回滚**: 健康检查失败时自动回滚

#### 配置示例
```yaml
# 手动部署到生产环境
on:
  workflow_dispatch:
    inputs:
      environment:
        description: '部署环境'
        required: true
        default: 'staging'
        type: choice
        options:
        - staging
        - production
```

### 3. 发布工作流 (`.github/workflows/release.yml`)

#### 触发条件
- 手动触发（指定版本号）
- 推送标签（格式: `v*`）

#### 发布流程
1. **创建GitHub发布版本**
2. **构建和标记Docker镜像**
3. **创建发布包**
4. **发布验证**
5. **发送通知**

#### 版本命名约定
- **正式版本**: `v1.0.0`, `v2.3.1`
- **预发布版本**: `v1.0.0-beta.1`, `v2.0.0-rc.2`
- **开发版本**: `v1.1.0-dev`

## 💻 本地开发集成

### 预提交钩子 (pre-commit)

安装预提交钩子，在提交前自动运行代码检查：

```bash
# 安装pre-commit
pip install pre-commit

# 安装git钩子
pre-commit install

# 手动运行所有钩子
pre-commit run --all-files
```

### `.pre-commit-config.yaml` 配置

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
        args: [--max-line-length=127]
  
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-json
```

### 本地Docker构建

```bash
# 构建Docker镜像
docker build -t self-soul-b:local .

# 运行本地测试
docker run --rm self-soul-b:local pytest tests/

# 运行完整CI流程
docker run --rm self-soul-b:local /bin/bash -c "
  black --check .
  flake8 .
  pytest tests/ --cov=core
"
```

## 🌍 环境管理

### 环境配置

#### 1. 测试环境 (staging)
```bash
# 环境变量
ENVIRONMENT=staging
DEBUG=true
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/self_soul_test
LOG_LEVEL=DEBUG
```

#### 2. 生产环境 (production)
```bash
# 环境变量
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://prod_user:${DB_PASSWORD}@prod-db:5432/self_soul_prod
LOG_LEVEL=INFO
SENTRY_DSN=https://sentry.io/your-project
PROMETHEUS_ENABLED=true
```

### 环境特定配置

创建环境特定的配置文件：

```bash
# docker-compose.staging.yml
version: '3.8'
services:
  self-soul-backend:
    environment:
      - ENVIRONMENT=staging
      - DEBUG=true
    ports:
      - "8000:8000"
  
  # 测试环境特定服务
  test-database:
    image: postgres:15
    environment:
      - POSTGRES_DB=self_soul_test
```

```bash
# docker-compose.production.yml  
version: '3.8'
services:
  self-soul-backend:
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    restart: always
    deploy:
      resources:
        limits:
          memory: 4G
  
  # 生产环境监控
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
```

### 环境变量管理

使用GitHub Secrets管理敏感环境变量：

```bash
# 设置GitHub Secrets
# 1. 在GitHub仓库设置中添加secrets
# 2. 在工作流中引用

# 示例：设置数据库密码
${{ secrets.PRODUCTION_DB_PASSWORD }}

# 示例：设置SSH密钥
${{ secrets.PRODUCTION_SSH_KEY }}
```

## 🔒 安全最佳实践

### 1. 安全扫描集成

#### Trivy漏洞扫描
```yaml
- name: 运行Trivy漏洞扫描
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: 'fs'
    scan-ref: '.'
    format: 'sarif'
```

#### Bandit安全扫描
```yaml
- name: 运行Bandit安全扫描
  run: |
    pip install bandit
    bandit -r core/ -f json -o bandit-report.json
```

### 2. 密钥管理

#### GitHub Secrets最佳实践
```yaml
# 不要硬编码密钥
# ❌ 错误做法
password: "my-secret-password"

# ✅ 正确做法  
password: ${{ secrets.API_PASSWORD }}
```

#### 临时密钥
```yaml
# 使用临时访问令牌
- name: 生成临时访问令牌
  run: |
    TEMP_TOKEN=$(aws sts assume-role ...)
    echo "TEMP_TOKEN=$TEMP_TOKEN" >> $GITHUB_ENV
```

### 3. 最小权限原则

#### 工作流权限
```yaml
permissions:
  contents: read
  packages: write
  # 仅授予必要权限
```

#### 容器安全
```yaml
# 使用非root用户运行容器
USER 1000:1000

# 只读文件系统
read_only: true

# 安全上下文
security_opt:
  - no-new-privileges:true
```

## 🔍 故障排除

### 常见问题

#### 1. CI流水线失败

**问题**: 代码质量检查失败
```bash
# 解决方案：本地运行代码检查
black .
flake8 .
mypy core/
```

**问题**: 测试失败
```bash
# 解决方案：本地运行失败测试
pytest tests/ -v -x

# 查看详细错误信息
pytest tests/ -v --tb=long
```

#### 2. 部署失败

**问题**: Docker构建失败
```bash
# 解决方案：本地测试构建
docker build -t test .

# 查看构建日志
docker build -t test . --progress=plain
```

**问题**: 健康检查失败
```bash
# 解决方案：手动检查服务
curl http://localhost:8000/api/health

# 查看容器日志
docker logs self-soul-backend
```

#### 3. 环境配置问题

**问题**: 环境变量缺失
```bash
# 解决方案：检查环境变量
echo $ENVIRONMENT
echo $DATABASE_URL

# 验证配置文件
cat .env.production
```

### 调试工具

#### GitHub Actions调试
```bash
# 启用调试日志
# 在仓库设置中添加secret
# ACTIONS_STEP_DEBUG = true
# ACTIONS_RUNNER_DEBUG = true

# 查看工作流运行日志
# https://github.com/OWNER/REPO/actions/runs/RUN_ID
```

#### 本地模拟CI环境
```bash
# 使用act工具本地运行GitHub Actions
brew install act  # macOS
act -P ubuntu-latest=node:16-buster-slim
```

## 🚀 扩展和自定义

### 1. 添加新的测试类型

#### 添加端到端测试
```yaml
# .github/workflows/e2e.yml
name: E2E Tests
on: [workflow_dispatch]

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - name: 运行端到端测试
        run: |
          python tests/e2e/test_full_workflow.py
```

#### 添加负载测试
```yaml
# .github/workflows/load-test.yml
name: Load Tests
on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点运行

jobs:
  load-test:
    runs-on: ubuntu-latest
    steps:
      - name: 运行负载测试
        run: |
          locust -f tests/load_test.py --host=http://localhost:8000
```

### 2. 集成第三方服务

#### 集成SonarQube代码质量
```yaml
- name: SonarQube扫描
  uses: SonarSource/sonarqube-scan-action@master
  with:
    args: >
      -Dsonar.projectKey=self-soul-b
      -Dsonar.sources=.
      -Dsonar.host.url=${{ secrets.SONAR_HOST_URL }}
      -Dsonar.login=${{ secrets.SONAR_TOKEN }}
```

#### 集成Sentry错误跟踪
```yaml
- name: 发送部署信息到Sentry
  run: |
    curl -X POST \
      -H "Authorization: Bearer ${{ secrets.SENTRY_AUTH_TOKEN }}" \
      -H "Content-Type: application/json" \
      -d '{"version": "${{ github.sha }}", "environment": "${{ github.ref_name }}"}' \
      https://sentry.io/api/0/organizations/ORG/releases/
```

### 3. 多平台构建

#### 构建多架构Docker镜像
```yaml
- name: 设置QEMU
  uses: docker/setup-qemu-action@v3

- name: 构建多平台镜像
  uses: docker/build-push-action@v5
  with:
    platforms: linux/amd64,linux/arm64
    push: true
    tags: user/app:latest
```

### 4. 自定义通知

#### 自定义Slack通知
```yaml
- name: 自定义Slack通知
  if: always()
  run: |
    STATUS="${{ job.status }}"
    COLOR="good"
    
    if [[ "$STATUS" == "failure" ]]; then
      COLOR="danger"
    elif [[ "$STATUS" == "cancelled" ]]; then
      COLOR="warning"
    fi
    
    curl -X POST -H 'Content-type: application/json' \
      --data "{
        \"attachments\": [{
          \"color\": \"$COLOR\",
          \"title\": \"CI/CD状态: $STATUS\",
          \"text\": \"工作流: ${{ github.workflow }}\\n分支: ${{ github.ref }}\\n提交: ${{ github.sha }}\",
          \"ts\": $(date +%s)
        }]
      }" \
      ${{ secrets.SLACK_WEBHOOK_URL }}
```

## 📊 监控和指标

### CI/CD指标收集

#### 收集构建指标
```python
# 收集CI/CD性能指标
import json
import time

metrics = {
    "workflow_id": os.getenv("GITHUB_RUN_ID"),
    "duration": time.time() - start_time,
    "success": success,
    "test_coverage": coverage_percentage,
    "vulnerabilities_found": vuln_count
}

# 发送到监控系统
requests.post("https://metrics.example.com/ci-metrics", json=metrics)
```

#### 生成CI/CD报告
```bash
# 生成HTML报告
pytest --cov=core --cov-report=html

# 生成JSON报告
pytest --cov=core --cov-report=json:coverage.json

# 上传报告
curl -X POST -H "Content-Type: application/json" \
  -d @coverage.json \
  ${{ secrets.COVERAGE_SERVICE_URL }}
```

### 性能优化

#### 缓存优化
```yaml
# 缓存Python依赖
- name: 缓存Python包
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('requirements.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

# 缓存Docker层
cache-from: type=gha
cache-to: type=gha,mode=max
```

#### 并行执行
```yaml
# 并行运行测试
jobs:
  unit-tests:
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11']
    
  integration-tests:
    needs: unit-tests
```

## 📚 最佳实践总结

### 代码质量
1. **预提交钩子**: 本地运行代码检查
2. **自动化测试**: 覆盖核心功能
3. **持续集成**: 频繁合并和测试

### 部署安全
1. **蓝绿部署**: 最小化停机时间
2. **健康检查**: 部署前验证
3. **自动回滚**: 失败时自动恢复

### 监控运维
1. **CI/CD指标**: 跟踪构建和部署性能
2. **警报通知**: 及时发现问题
3. **文档更新**: 保持文档同步

### 团队协作
1. **标准化流程**: 统一工作流配置
2. **权限管理**: 最小权限原则
3. **知识共享**: 文档和培训

---

*本文档最后更新: 2026-03-06*
*版本: v1.0.0*

## 🔗 相关资源
- [GitHub Actions文档](https://docs.github.com/en/actions)
- [Docker文档](https://docs.docker.com/)
- [预提交钩子文档](https://pre-commit.com/)
- [安全最佳实践](https://securitylab.github.com/)