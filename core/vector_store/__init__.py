"""
向量存储模块

提供多模态向量存储和检索功能。
支持文本、图像、音频、视频等多种模态的嵌入存储和相似度检索。
"""

# 尝试导入向量存储后端，优雅处理导入失败
ChromaVectorStore = None
MultimodalVectorStore = None
MemoryVectorStore = None

try:
    from .chroma_vector_store import ChromaVectorStore
except Exception:
    ChromaVectorStore = None

try:
    from .multimodal_vector_store import MultimodalVectorStore
except Exception:
    MultimodalVectorStore = None

try:
    from .memory_vector_store import MemoryVectorStore
except Exception:
    MemoryVectorStore = None

__all__ = [
    "ChromaVectorStore",
    "MultimodalVectorStore",
    "MemoryVectorStore",
]