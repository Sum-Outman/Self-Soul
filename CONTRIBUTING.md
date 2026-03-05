# 贡献指南

感谢您有兴趣为Self-Soul-B多模态AGI系统做出贡献！本指南将帮助您了解如何为项目做出贡献。

## 📋 目录
- [行为准则](#行为准则)
- [开发流程](#开发流程)
- [开发环境设置](#开发环境设置)
- [代码规范](#代码规范)
- [测试要求](#测试要求)
- [文档要求](#文档要求)
- [提交规范](#提交规范)
- [Pull Request流程](#pull-request流程)
- [问题报告](#问题报告)
- [功能建议](#功能建议)

## 👥 行为准则

我们致力于为所有贡献者创造友好、尊重和包容的环境。请阅读并遵守我们的[行为准则](CODE_OF_CONDUCT.md)。

## 🔄 开发流程

### 1. 问题发现与讨论
- 在开始任何工作之前，请先检查[现有Issues](https://github.com/your-org/self-soul-b/issues)
- 如果是bug报告，请使用[bug报告模板](.github/ISSUE_TEMPLATE/bug_report.md)
- 如果是功能请求，请使用[功能请求模板](.github/ISSUE_TEMPLATE/feature_request.md)
- 如果是技术债务，请使用[技术债务模板](.github/ISSUE_TEMPLATE/technical_debt.md)

### 2. 分支策略
```
main (保护分支) - 生产环境代码
├── develop - 开发集成分支
├── feature/* - 功能开发分支
├── bugfix/* - bug修复分支
├── release/* - 发布准备分支
└── hotfix/* - 紧急修复分支
```

### 3. 分支命名规范
- 功能分支: `feature/简短描述-issue编号`，例如: `feature/multimodal-fusion-123`
- Bug修复分支: `bugfix/简短描述-issue编号`，例如: `bugfix/image-processing-456`
- 发布分支: `release/版本号`，例如: `release/v1.2.0`
- 紧急修复分支: `hotfix/简短描述`，例如: `hotfix/security-patch`

## 🛠️ 开发环境设置

### 前置要求
- Python 3.10+
- Node.js 16+ (前端开发需要)
- Docker & Docker Compose (可选)
- Git

### 后端环境设置
```bash
# 1. 克隆仓库
git clone https://github.com/your-org/self-soul-b.git
cd self-soul-b

# 2. 创建虚拟环境
python -m venv venv

# 3. 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. 安装依赖
pip install -r requirements.txt

# 5. 设置环境变量
cp .env.example .env
# 编辑.env文件配置必要的环境变量

# 6. 初始化数据库
python core/main.py --init
```

### 前端环境设置
```bash
cd app
npm install
npm run dev
```

### Docker开发环境
```bash
# 使用Docker Compose启动开发环境
docker-compose -f docker-compose.dev.yml up -d
```

## 📝 代码规范

### Python代码规范
- 遵循[PEP 8](https://pep8.org/)规范
- 使用[Black](https://black.readthedocs.io/)进行代码格式化
- 使用[isort](https://pycqa.github.io/isort/)进行导入排序
- 使用[flake8](https://flake8.pycqa.org/)进行代码检查

### 类型提示
- 所有函数和方法必须包含类型提示
- 使用Python 3.10+的类型提示语法

### 文档字符串
- 使用Google风格的文档字符串
- 所有公共函数、类和模块必须有文档字符串

### 示例代码
```python
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrueImageProcessor:
    """真实图像处理器
    
    Attributes:
        target_size: 目标图像尺寸
    """
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """初始化图像处理器
        
        Args:
            target_size: 目标图像尺寸
        """
        self.target_size = target_size
    
    def preprocess_image(self, image_data: Union[bytes, torch.Tensor]) -> torch.Tensor:
        """预处理图像
        
        Args:
            image_data: 图像数据（字节或张量）
            
        Returns:
            预处理后的图像张量
            
        Raises:
            ValueError: 如果无法处理图像数据
        """
        # 实现代码
```

## 🧪 测试要求

### 测试框架
- 使用[pytest](https://docs.pytest.org/)作为测试框架
- 测试文件命名: `test_*.py`
- 测试函数命名: `test_*`

### 测试覆盖率
- 目标覆盖率: >80%
- 新增代码必须有相应的测试
- 使用[pytest-cov](https://pytest-cov.readthedocs.io/)检查覆盖率

### 测试类型
1. **单元测试**: 测试单个函数或类的功能
2. **集成测试**: 测试模块间的集成
3. **端到端测试**: 测试完整的工作流程
4. **性能测试**: 测试系统性能指标

### 测试示例
```python
import pytest
from core.multimodal.true_data_processor import TrueImageProcessor


class TestTrueImageProcessor:
    """测试TrueImageProcessor类"""
    
    def setup_method(self):
        """测试设置"""
        self.processor = TrueImageProcessor()
    
    def test_preprocess_image_with_bytes(self):
        """测试使用字节数据预处理图像"""
        # 测试代码
    
    def test_preprocess_image_with_tensor(self):
        """测试使用张量数据预处理图像"""
        # 测试代码
    
    def test_invalid_image_data(self):
        """测试无效图像数据"""
        with pytest.raises(ValueError):
            self.processor.preprocess_image(None)
```

## 📚 文档要求

### 文档类型
1. **API文档**: 使用[Sphinx](https://www.sphinx-doc.org/)生成
2. **用户手册**: Markdown格式，位于`docs/`目录
3. **开发文档**: 代码中的文档字符串和注释
4. **部署文档**: Docker和运维相关文档

### 文档更新
- 代码变更必须更新相关文档
- 新增功能必须提供使用示例
- API变更必须更新API文档

## 💬 提交规范

### 提交信息格式
```
类型(范围): 简短描述

详细描述（可选）

关联Issue: #123
```

### 提交类型
- `feat`: 新功能
- `fix`: bug修复
- `docs`: 文档更新
- `style`: 代码格式（不影响功能）
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具变更

### 提交示例
```
feat(multimodal): 添加真实图像处理流水线

- 实现TrueImageProcessor类，支持多种图像格式
- 添加图像格式检测功能
- 集成OpenCV和PIL进行图像解码

关联Issue: #123
```

## 🔀 Pull Request流程

### PR创建要求
1. 从`develop`分支创建功能分支
2. 完成功能开发并提交测试
3. 确保所有测试通过
4. 更新相关文档
5. 创建Pull Request

### PR审查流程
1. **代码审查**: 至少需要2个核心贡献者批准
2. **CI检查**: 所有CI检查必须通过
3. **测试覆盖率**: 新代码必须有足够的测试覆盖
4. **文档检查**: 相关文档必须更新
5. **最终合并**: 由项目维护者合并

### PR模板
请使用[PR模板](.github/PULL_REQUEST_TEMPLATE.md)

## 🐛 问题报告

### 报告要求
1. 使用[bug报告模板](.github/ISSUE_TEMPLATE/bug_report.md)
2. 提供详细的复现步骤
3. 包含环境信息和日志
4. 描述预期行为和实际行为

### 严重性评估
- **P0 (紧急)**: 系统完全不可用或数据丢失
- **P1 (高)**: 核心功能无法使用
- **P2 (中)**: 次要功能问题
- **P3 (低)**: 界面优化或小问题

## 💡 功能建议

### 建议要求
1. 使用[功能请求模板](.github/ISSUE_TEMPLATE/feature_request.md)
2. 详细描述使用场景和需求
3. 提供技术实现建议
4. 评估优先级和影响

### 功能优先级
- **P0**: 战略级功能，直接影响产品竞争力
- **P1**: 核心功能增强，显著提升用户体验
- **P2**: 优化功能，提升系统性能或易用性
- **P3**: 边缘功能，可有可无的增强

## 📞 获取帮助

### 沟通渠道
- [GitHub Discussions](https://github.com/your-org/self-soul-b/discussions): 一般讨论
- [GitHub Issues](https://github.com/your-org/self-soul-b/issues): bug报告和功能请求
- [Slack/Discord](链接): 实时交流

### 寻求帮助
- 在开始复杂功能开发前，建议先进行讨论
- 如果遇到技术难题，可以创建技术讨论issue
- 对于新手，可以从标记为`good-first-issue`的问题开始

## 🙏 致谢

感谢所有为Self-Soul-B项目做出贡献的开发者！您的每一份贡献都让项目变得更好。

---

*本贡献指南最后更新: 2026-03-06*