# Self-Soul-B多模态AGI系统API集成指南

本文档提供了如何配置和集成真实生成模型API到Self-Soul-B多模态AGI系统的详细指南。

## 📋 目录
- [支持的API提供商](#支持的api提供商)
- [配置方法](#配置方法)
- [连接测试](#连接测试)
- [使用示例](#使用示例)
- [故障排除](#故障排除)
- [安全注意事项](#安全注意事项)

## 🏢 支持的API提供商

Self-Soul-B系统支持以下生成模型API提供商：

### 核心提供商
| 提供商 | 支持的服务 | 默认模型 | Python包 |
|--------|------------|----------|----------|
| **OpenAI** | Chat, Vision, Embeddings | gpt-4o | `openai` |
| **Anthropic** | Chat, Claude | claude-3-opus-20240229 | `anthropic` |
| **Google AI** | Gemini, PaLM | gemini-pro | `google-generativeai` |
| **AWS Bedrock** | 多种模型 | anthropic.claude-3-sonnet | `boto3` |
| **Azure OpenAI** | OpenAI兼容 | gpt-4 | `openai` |

### 国内提供商
| 提供商 | 支持的服务 | 默认模型 | Python包 |
|--------|------------|----------|----------|
| **智谱AI (Zhipu)** | ChatGLM | glm-4 | `zhipuai` |
| **百度文心 (ERNIE)** | 文心一言 | ernie-bot-4.0 | `erniebot` |
| **阿里通义千问** | Qwen | qwen-plus | `dashscope` |
| **讯飞星火** | Spark | v3.0 | `xfyun` |
| **腾讯混元** | Hunyuan | hunyuan-standard | `tencentcloud-sdk-python` |

### 开源/自托管
| 提供商 | 支持的服务 | 默认模型 | Python包 |
|--------|------------|----------|----------|
| **HuggingFace** | 推理端点 | 自定义 | `huggingface_hub` |
| **Ollama** | 本地LLM | llama2 | `ollama` |
| **vLLM** | 高性能推理 | 自定义 | `vllm` |

## 🔧 配置方法

### 方法1：环境变量配置（推荐）

创建`.env`文件在项目根目录：

```bash
# OpenAI配置
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，用于自定义端点
OPENAI_MODEL=gpt-4o

# Anthropic配置
ANTHROPIC_API_KEY=your-anthropic-api-key-here
ANTHROPIC_MODEL=claude-3-opus-20240229

# Google AI配置
GOOGLE_API_KEY=your-google-api-key-here
GOOGLE_MODEL=gemini-pro

# 智谱AI配置
ZHIPUAI_API_KEY=your-zhipuai-api-key-here
ZHIPUAI_MODEL=glm-4

# 百度文心配置
BAIDU_API_KEY=your-baidu-api-key-here
BAIDU_SECRET_KEY=your-baidu-secret-key-here

# 其他通用配置
API_TIMEOUT=60
API_MAX_RETRIES=3
API_TEMPERATURE=0.7
```

### 方法2：配置文件配置

编辑`core/data/settings/system_settings.json`中的`external_api_configs`部分：

```json
{
  "external_api_configs": {
    "openai_config": {
      "name": "OpenAI API",
      "api_type": "openai",
      "api_url": "https://api.openai.com/v1",
      "api_key": "sk-your-actual-api-key-here",
      "model_name": "gpt-4o",
      "description": "OpenAI GPT-4o API",
      "enabled": true,
      "default": true
    },
    "anthropic_config": {
      "name": "Anthropic Claude",
      "api_type": "anthropic",
      "api_url": "https://api.anthropic.com",
      "api_key": "your-anthropic-api-key-here",
      "model_name": "claude-3-opus-20240229",
      "description": "Anthropic Claude 3 Opus",
      "enabled": true,
      "default": false
    }
  }
}
```

### 方法3：运行时配置

通过API动态配置：

```python
from core.external_api_service import ExternalAPIService

api_service = ExternalAPIService()

# 添加OpenAI配置
api_service.configure_api_provider("openai", {
    "api_key": "sk-your-api-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o"
})

# 测试连接
result = api_service.test_connection("openai")
print(f"连接测试结果: {result}")
```

## 🔌 连接测试

### 1. 使用测试脚本

运行API连接测试脚本：

```bash
# 测试所有已配置的API
python tests/test_all_apis.py

# 测试特定API提供商
python tests/test_api_connection.py --provider openai
python tests/test_api_connection.py --provider anthropic
python tests/test_api_connection.py --provider google
```

### 2. 使用Python代码测试

```python
import sys
sys.path.insert(0, '.')

from core.api_config_manager import APIConfigManager
from core.api_model_connector import api_model_connector

# 方法1: 使用APIConfigManager
config_manager = APIConfigManager()
result = config_manager.test_api_connection("openai")
print(f"OpenAI连接测试: {result}")

# 方法2: 使用APIModelConnector
result = api_model_connector.test_api_connection("openai", {
    "api_url": "https://api.openai.com/v1",
    "api_key": "sk-your-api-key",
    "model_name": "gpt-3.5-turbo"
})
print(f"API模型连接测试: {result}")
```

### 3. 使用REST API测试

启动系统后，使用以下API端点：

```bash
# 测试所有API连接
curl -X POST http://localhost:8000/api/external-api/test-all

# 测试特定API
curl -X POST http://localhost:8000/api/external-api/test/openai_config \
  -H "Content-Type: application/json" \
  -d '{"test_input": "Hello"}'
```

## 🚀 使用示例

### 1. 文本生成

```python
from core.external_api_service import ExternalAPIService

api_service = ExternalAPIService()

# 使用OpenAI生成文本
response = api_service.generate_text(
    provider="openai",
    prompt="写一首关于春天的诗",
    model="gpt-4o",
    temperature=0.8,
    max_tokens=500
)
print(f"生成的文本: {response}")

# 使用多个提供商（自动选择最佳）
response = api_service.generate_text_with_fallback(
    prompt="解释量子计算的基本原理",
    preferred_providers=["openai", "anthropic", "google"],
    fallback_to_local=True  # 如果所有API都失败，回退到本地模型
)
```

### 2. 多模态生成

```python
# 图像描述（文本到图像理解）
image_url = "https://example.com/image.jpg"
description = api_service.describe_image(
    provider="openai",
    image_url=image_url,
    prompt="描述这张图片中的内容",
    model="gpt-4-vision-preview"
)

# 图像生成（文本到图像）
image_data = api_service.generate_image(
    provider="openai",
    prompt="一只可爱的猫咪在花园里玩耍",
    model="dall-e-3",
    size="1024x1024"
)

# 保存生成的图像
with open("generated_image.png", "wb") as f:
    f.write(image_data)
```

### 3. 语音处理

```python
# 语音转文本
audio_file = "speech.wav"
transcript = api_service.transcribe_audio(
    provider="openai",
    audio_file=audio_file,
    model="whisper-1"
)

# 文本转语音
audio_data = api_service.synthesize_speech(
    provider="openai",
    text="你好，我是AI助手",
    model="tts-1",
    voice="alloy"
)

# 保存语音文件
with open("speech_output.mp3", "wb") as f:
    f.write(audio_data)
```

### 4. 多模态融合

```python
# 结合文本和图像的多模态处理
multimodal_response = api_service.process_multimodal(
    inputs={
        "text": "这张图片显示的是什么？",
        "image": "image.jpg"
    },
    task="visual_question_answering",
    providers={
        "vision": "openai",
        "language": "anthropic"
    }
)
```

## 🔍 故障排除

### 常见问题

#### 1. API密钥无效
**症状**: 返回401错误
**解决方案**:
- 检查API密钥是否正确
- 确保API密钥有足够的权限
- 对于OpenAI，检查是否设置了正确的组织ID（如果需要）

#### 2. 连接超时
**症状**: 请求超时
**解决方案**:
- 检查网络连接
- 增加超时设置：`API_TIMEOUT=120`
- 使用代理（如果需要）：`export HTTPS_PROXY=http://proxy:port`

#### 3. 速率限制
**症状**: 返回429错误
**解决方案**:
- 降低请求频率
- 实现指数退避重试
- 升级API套餐以获得更高限制

#### 4. 模型不可用
**症状**: 返回404或模型未找到错误
**解决方案**:
- 检查模型名称是否正确
- 确保您的API密钥有权访问该模型
- 尝试使用更通用的模型

### 调试步骤

1. **检查基本连接**:
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

2. **检查Python包版本**:
   ```bash
   pip show openai anthropic google-generativeai
   ```

3. **启用详细日志**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **使用测试模式**:
   ```python
   # 启用测试模式，不发送真实请求
   os.environ["API_TEST_MODE"] = "true"
   ```

## 🔒 安全注意事项

### API密钥安全
1. **永远不要提交API密钥到版本控制**
   - 确保`.env`在`.gitignore`中
   - 使用环境变量而不是硬编码

2. **密钥轮换**
   - 定期更换API密钥
   - 使用密钥管理服务（如AWS Secrets Manager）

3. **权限最小化**
   - 只为必要操作授予API密钥权限
   - 使用只读密钥（如果适用）

### 数据隐私
1. **敏感数据**
   - 不要通过API发送敏感个人信息
   - 考虑使用本地模型处理敏感数据

2. **数据保留**
   - 了解API提供商的数据保留政策
   - 启用数据删除功能（如果可用）

3. **合规性**
   - 确保符合GDPR、CCPA等法规
   - 了解数据跨境传输规定

### 成本控制
1. **使用限制**
   ```python
   # 设置使用限制
   os.environ["API_MAX_TOKENS_PER_DAY"] = "1000000"
   os.environ["API_MAX_REQUESTS_PER_MINUTE"] = "60"
   ```

2. **监控使用量**
   - 定期检查API使用报告
   - 设置使用量告警

3. **成本优化**
   - 使用更便宜的模型进行简单任务
   - 缓存API响应
   - 批量处理请求

## 📊 性能优化

### 1. 连接池
```python
# 启用连接池
os.environ["API_ENABLE_CONNECTION_POOL"] = "true"
os.environ["API_POOL_SIZE"] = "10"
```

### 2. 请求批处理
```python
# 批量处理多个请求
responses = api_service.batch_generate_text(
    prompts=["提示1", "提示2", "提示3"],
    provider="openai",
    batch_size=10
)
```

### 3. 响应缓存
```python
# 启用响应缓存
os.environ["API_ENABLE_CACHE"] = "true"
os.environ["API_CACHE_TTL_SECONDS"] = "3600"  # 1小时
```

### 4. 异步处理
```python
import asyncio

async def process_multiple_apis():
    tasks = [
        api_service.async_generate_text("提示1", provider="openai"),
        api_service.async_generate_text("提示2", provider="anthropic"),
        api_service.async_generate_text("提示3", provider="google")
    ]
    
    responses = await asyncio.gather(*tasks)
    return responses
```

## 🧪 测试策略

### 单元测试
```python
# tests/test_api_integration.py
def test_openai_integration():
    """测试OpenAI API集成"""
    # 使用模拟响应进行测试
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = {"choices": [{"message": {"content": "模拟响应"}}]}
        
        response = api_service.generate_text("测试提示", provider="openai")
        assert response == "模拟响应"
```

### 集成测试
```python
def test_api_fallback_mechanism():
    """测试API回退机制"""
    # 模拟第一个API失败
    with patch('openai.ChatCompletion.create', side_effect=Exception("API失败")):
        # 应该回退到第二个API
        response = api_service.generate_text_with_fallback(
            "测试提示",
            preferred_providers=["openai", "anthropic"]
        )
        assert response is not None
```

### 性能测试
```python
def test_api_performance():
    """测试API性能"""
    import time
    
    start_time = time.time()
    for i in range(10):
        api_service.generate_text(f"测试提示{i}")
    
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / 10
    
    assert avg_time < 2.0  # 平均响应时间应小于2秒
```

## 📞 支持与帮助

### 获取帮助
- **GitHub Issues**: 报告问题和请求功能
- **文档**: 查看详细API文档
- **社区**: 加入开发者社区讨论

### 贡献指南
如果您想为API集成添加新的提供商，请参考[贡献指南](CONTRIBUTING.md)。

### 更新日志
API集成功能的更新记录在[CHANGELOG.md](CHANGELOG.md)中。

---

*最后更新: 2026-03-06*
*版本: v1.0.0*