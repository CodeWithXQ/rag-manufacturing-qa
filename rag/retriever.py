"""检索模块（改进点核心）：向量检索 + BM25 检索 + RRF 融合 + rerank 精排。

- 纯向量检索（baseline）：FAISS 直接返回 top-k
- 混合检索（改进）：向量 top-k1 + BM25 top-k2 → RRF 融合去重 → bge-reranker 精排
"""
from __future__ import annotations

import numpy as np
from sentence_transformers import CrossEncoder

from config import (BM25_TOP_K, FINAL_TOP_K, RERANK_MODEL_PATH, RRF_K,
                    VECTOR_TOP_K)
from rag.indexer import Index, Indexer, tokenize
from rag.loader import Chunk


class Retriever:
    def __init__(self, index: Index, indexer: Indexer,
                 rerank_model_path=RERANK_MODEL_PATH):
        self.index = index
        self.indexer = indexer
        self.rerank_model = CrossEncoder(str(rerank_model_path))

    def vector_search(self, query: str, top_k: int = VECTOR_TOP_K):
        q = self.indexer.embed_query(query)
        scores, ids = self.index.faiss_index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.index.chunks[idx], float(score)))
        return results

    def bm25_search(self, query: str, top_k: int = BM25_TOP_K):
        toks = tokenize(query)
        scores = self.index.bm25.get_scores(toks)
        top_ids = np.argsort(scores)[::-1][:top_k]
        return [(self.index.chunks[i], float(scores[i]))
                for i in top_ids if scores[i] > 0]

    def hybrid_search(self, query: str, vector_top_k: int = VECTOR_TOP_K,
                      bm25_top_k: int = BM25_TOP_K,
                      final_top_k: int = FINAL_TOP_K):
        v_res = self.vector_search(query, vector_top_k)
        b_res = self.bm25_search(query, bm25_top_k)
        merged = self._rrf_fusion(v_res, b_res, RRF_K)
        if not merged:
            return []
        # RRF 融合后限制候选数，减少 rerank 引入的噪声
        merged = merged[:max(final_top_k * 3, 10)]
        chunks = [c for c, _ in merged]
        rerank_scores = self.rerank_model.predict(
            [(query, c.text) for c in chunks])
        order = np.argsort(rerank_scores)[::-1][:final_top_k]
        return [(chunks[i], float(rerank_scores[i])) for i in order]

    @staticmethod
    def _rrf_fusion(v_res, b_res, k: int = RRF_K):
        chunk_map: dict[str, Chunk] = {}
        rrf: dict[str, float] = {}
        for rank, (chunk, _) in enumerate(v_res, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        for rank, (chunk, _) in enumerate(b_res, start=1):
            chunk_map[chunk.chunk_id] = chunk
            rrf[chunk.chunk_id] = rrf.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
        sorted_ids = sorted(rrf, key=rrf.get, reverse=True)
        return [(chunk_map[cid], rrf[cid]) for cid in sorted_ids]

    def search(self, query: str, mode: str = "hybrid",
               top_k: int = FINAL_TOP_K):
        if mode == "vector":
            return self.vector_search(query, top_k)
        if mode == "hybrid":
            return self.hybrid_search(query, final_top_k=top_k)
        raise ValueError(f"未知检索模式: {mode}")
