"""召回率对比实验：纯向量检索 vs 混合检索，chunk 级 Recall@k。

评估口径（比文档级更严格）：检索前 k 个文本块中，命中"标准答案所在章节"
（即 gold_doc 的「排查步骤」或「常见原因」章节）才算召回成功。
用法：python evaluate.py
"""
from __future__ import annotations

import json

from config import FINAL_TOP_K, INDEX_DIR, TESTSET_DIR
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"


def load_testset() -> list[dict]:
    return json.loads(
        (TESTSET_DIR / "testset.json").read_text(encoding="utf-8"))


def gold_section_keyword(question: str) -> str:
    # 问"原因"类，标准答案在「常见原因」章节；其余在「排查步骤」章节
    return "常见原因" if "原因" in question else "排查步骤"


def chunk_hit(results, gold_doc: str, gold_keyword: str) -> bool:
    for chunk, _ in results:
        if chunk.source == gold_doc and gold_keyword in chunk.text:
            return True
    return False


def recall_at_k(retriever: Retriever, testset: list[dict],
                mode: str, k: int) -> float:
    hit = 0
    for item in testset:
        results = retriever.search(item["question"], mode=mode, top_k=k)
        kw = gold_section_keyword(item["question"])
        if chunk_hit(results, item["gold_doc"], kw):
            hit += 1
    return hit / len(testset)


def mrr(retriever: Retriever, testset: list[dict], mode: str,
        k: int = FINAL_TOP_K) -> float:
    """Mean Reciprocal Rank：标准答案段落首次出现排名的倒数均值，衡量排序质量。"""
    total = 0.0
    for item in testset:
        results = retriever.search(item["question"], mode=mode, top_k=k)
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

    print(f"测试集：{len(testset)} 个问题（chunk 级评估）")
    print(f"{'指标':<14}{'纯向量':>12}{'混合检索':>12}{'提升':>10}")
    print("-" * 50)
    for k in [1, 3, 5]:
        vec = recall_at_k(retriever, testset, "vector", k)
        hyb = recall_at_k(retriever, testset, "hybrid", k)
        print(f"{'Recall@%d' % k:<14}{vec:>12.2%}{hyb:>12.2%}{hyb - vec:>+10.2%}")
    vec_mrr = mrr(retriever, testset, "vector")
    hyb_mrr = mrr(retriever, testset, "hybrid")
    print(f"{'MRR':<14}{vec_mrr:>12.4f}{hyb_mrr:>12.4f}{hyb_mrr - vec_mrr:>+10.4f}")

    print(f"\n逐题对比（Recall@{FINAL_TOP_K}，✓=命中标准答案章节，✗=未命中）：")
    print(f"{'ID':<4}{'问题':<36}{'纯向量':<8}{'混合':<8}")
    for item in testset:
        kw = gold_section_keyword(item["question"])
        vec_ok = chunk_hit(
            retriever.search(item["question"], "vector", top_k=FINAL_TOP_K),
            item["gold_doc"], kw)
        hyb_ok = chunk_hit(
            retriever.search(item["question"], "hybrid", top_k=FINAL_TOP_K),
            item["gold_doc"], kw)
        print(f"{item['id']:<4}{item['question']:<36}"
              f"{'✓' if vec_ok else '✗':<8}{'✓' if hyb_ok else '✗':<8}")


if __name__ == "__main__":
    main()
