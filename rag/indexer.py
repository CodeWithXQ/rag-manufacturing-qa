"""建库：embedding + FAISS 向量索引 + BM25 关键词索引。

- embedding 用 bge-large-zh-v1.5（本地加载，不烧 API）
- 向量索引用 FAISS IndexFlatIP（向量已 L2 归一化，内积即余弦相似度）
- 关键词索引用 rank_bm25 的 BM25Okapi，中文用 jieba 分词
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import BGE_QUERY_PREFIX, EMBED_MODEL_PATH
from rag.loader import Chunk


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


@dataclass
class Index:
    """建库产物：向量索引 + 关键词索引 + 原始块（不含 embedding 模型）。"""
    faiss_index: object
    bm25: BM25Okapi
    chunks: list[Chunk]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "Index":
        with open(path, "rb") as f:
            return pickle.load(f)


class Indexer:
    def __init__(self, embed_model_path: Path | str = EMBED_MODEL_PATH):
        self.embed_model = SentenceTransformer(str(embed_model_path))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        # passage 不加前缀（bge 规范：只有 query 加前缀）
        vecs = self.embed_model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def embed_query(self, query: str) -> np.ndarray:
        # query 加前缀
        return np.asarray(
            self.embed_model.encode(
                [BGE_QUERY_PREFIX + query], normalize_embeddings=True),
            dtype="float32")

    def build(self, chunks: list[Chunk]) -> Index:
        texts = [c.text for c in chunks]
        vectors = self.embed_passages(texts)

        dim = vectors.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(vectors)

        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)

        return Index(faiss_index, bm25, chunks)
