"""消融实验：只用向量 / 只用 BM25 / 混合检索 三组对比（chunk 级 Recall@k + MRR）。

回答面试高频问题"各部分贡献多少"：
  - vector：纯向量检索（FAISS + bge-large-zh）
  - bm25  ：纯关键词检索（rank_bm25 + jieba）
  - hybrid：向量 + BM25 → RRF 融合 → bge-reranker 精排
用法：python ablation.py
"""
from __future__ import annotations

import json

from config import FINAL_TOP_K, INDEX_DIR, TESTSET_DIR
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"
MODES = ["vector", "bm25", "hybrid"]


def load_testset() -> list[dict]:
    return json.loads(
        (TESTSET_DIR / "testset.json").read_text(encoding="utf-8"))


def gold_section_keyword(question: str) -> str:
    return "常见原因" if "原因" in question else "排查步骤"


def chunk_hit(results, gold_doc: str, gold_keyword: str) -> bool:
    for chunk, _ in results:
        if chunk.source == gold_doc and gold_keyword in chunk.text:
            return True
    return False


def search_by_mode(retriever, query, mode, top_k):
    if mode == "vector":
        return retriever.vector_search(query, top_k)
    if mode == "bm25":
        return retriever.bm25_search(query, top_k)
    if mode == "hybrid":
        return retriever.hybrid_search(query, final_top_k=top_k)
    raise ValueError(mode)


def recall_at_k(retriever, testset, mode, k) -> float:
    hit = 0
    for item in testset:
        results = search_by_mode(retriever, item["question"], mode, k)
        kw = gold_section_keyword(item["question"])
        if chunk_hit(results, item["gold_doc"], kw):
            hit += 1
    return hit / len(testset)


def mrr(retriever, testset, mode, k=FINAL_TOP_K) -> float:
    total = 0.0
    for item in testset:
        results = search_by_mode(retriever, item["question"], mode, k)
        kw = gold_section_keyword(item["question"])
        for rank, (chunk, _) in enumerate(results, start=1):
            if chunk.source == item["gold_doc"] and kw in chunk.text:
                total += 1.0 / rank
                break
    return total / len(testset)


def main() -> None:
    index = Index.load(INDEX_PATH)
    indexer = Indexer()
    retriever = Retriever(index, indexer)
    testset = load_testset()

    print(f"Testset: {len(testset)} questions (chunk-level)")
    print(f"{'Metric':<12}{'vector':>10}{'bm25':>10}{'hybrid':>10}")
    print("-" * 44)

    recalls = {m: {} for m in MODES}
    for k in [1, 3, 5]:
        for m in MODES:
            recalls[m][k] = recall_at_k(retriever, testset, m, k)
        print(f"{'Recall@%d' % k:<12}"
              f"{recalls['vector'][k]:>10.2%}"
              f"{recalls['bm25'][k]:>10.2%}"
              f"{recalls['hybrid'][k]:>10.2%}")

    mrrs = {m: mrr(retriever, testset, m) for m in MODES}
    print(f"{'MRR':<12}{mrrs['vector']:>10.4f}{mrrs['bm25']:>10.4f}{mrrs['hybrid']:>10.4f}")

    print("\n--- Delta (hybrid - X) ---")
    print(f"Recall@1:  vs vector {recalls['hybrid'][1] - recalls['vector'][1]:+.2%}"
          f"  vs bm25 {recalls['hybrid'][1] - recalls['bm25'][1]:+.2%}")
    print(f"MRR:       vs vector {mrrs['hybrid'] - mrrs['vector']:+.4f}"
          f"  vs bm25 {mrrs['hybrid'] - mrrs['bm25']:+.4f}")


if __name__ == "__main__":
    main()
