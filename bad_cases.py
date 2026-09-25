"""Bad case 分析：找出混合检索 Recall@5 未命中的题，打印 top5 检索结果辅助定位失败原因。

用法：python bad_cases.py
"""
from __future__ import annotations

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

from config import FINAL_TOP_K, INDEX_DIR, TESTSET_DIR
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"


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


def main() -> None:
    index = Index.load(INDEX_PATH)
    indexer = Indexer()
    retriever = Retriever(index, indexer)
    testset = load_testset()

    misses = []
    for item in testset:
        results = retriever.hybrid_search(item["question"], final_top_k=FINAL_TOP_K)
        kw = gold_section_keyword(item["question"])
        if not chunk_hit(results, item["gold_doc"], kw):
            misses.append((item, results, kw))

    print(f"共 {len(testset)} 题，混合检索 Recall@{FINAL_TOP_K} 未命中 {len(misses)} 题\n")

    for item, results, kw in misses:
        print("=" * 72)
        print(f"[{item['id']}] 问题：{item['question']}")
        print(f"标准答案：{item['gold_doc']}（目标章节：{kw}）")
        print(f"检索 top{FINAL_TOP_K}：")
        for rank, (chunk, score) in enumerate(results, start=1):
            star = "★" if chunk.source == item["gold_doc"] else " "
            snippet = chunk.text.replace("\n", " ")[:68]
            print(f"  {rank}. {star} [{chunk.source}] {snippet}")
        print()


if __name__ == "__main__":
    main()
