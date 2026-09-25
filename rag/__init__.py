"""制造设备故障知识库问答 Agent（RAG）核心包。

模块划分：
- loader     文档读取 + 切分
- indexer    embedding + FAISS 向量索引 + BM25 关键词索引
- retriever  向量检索 + BM25 检索 + RRF 融合 + rerank 精排（改进点）
- generator  prompt 拼接 + LLM 生成 + 引用解析
- llm        统一 LLM 接入层（OpenAI 兼容协议）
"""
__version__ = "0.1.0"
