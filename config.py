"""全局配置：路径、检索参数、LLM 接入，均从环境变量 / .env 读取。

核心 RAG 链路（切分 / embedding / 检索 / 融合 / 生成）全部自己实现，不依赖
LangChain 等重框架，便于讲清每一步原理。
"""
from __future__ import annotations

import os
from pathlib import Path

# --- 项目路径 ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
TESTSET_DIR = DATA_DIR / "testset"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = PROJECT_ROOT / "index_store"

# --- 文档切分参数 ---
CHUNK_SIZE = 512          # 每块最大字符数
CHUNK_OVERLAP = 64        # 相邻块重叠字符数

# --- 检索参数 ---
VECTOR_TOP_K = 20         # 向量检索召回数
BM25_TOP_K = 20           # BM25 关键词检索召回数
RRF_K = 60                # RRF 融合常数
FINAL_TOP_K = 5           # rerank 精排后最终返回块数

# --- 模型路径（download_models.py 下载到本地）---
EMBED_MODEL_PATH = MODELS_DIR / "bge-large-zh-v1.5"
RERANK_MODEL_PATH = MODELS_DIR / "bge-reranker-base"

# bge 系列规范：query 需要加前缀，passage 不加（见 BAAI/bge 官方 README）
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def load_env(path: Path | None = None) -> None:
    """从 .env 读取 key=value 到 os.environ（已存在的变量不覆盖）。"""
    env_path = path or (PROJECT_ROOT / ".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_llm_config() -> dict:
    """返回 LLM 接入配置（OpenAI 兼容协议，DeepSeek / 通义 / 智谱通用）。"""
    load_env()
    return {
        "base_url": os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
        "api_key": os.environ.get("LLM_API_KEY", ""),
        "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
        "timeout": int(os.environ.get("LLM_TIMEOUT", "60")),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "1024")),
    }
