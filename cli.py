"""命令行问答（支持单次 + 交互模式）。

用法：
  python cli.py "注塑机飞边怎么排查"              # 单次，默认混合检索
  python cli.py "注塑机飞边怎么排查" --mode vector  # 纯向量检索
  python cli.py                                    # 交互模式
"""
from __future__ import annotations

import argparse

from config import INDEX_DIR
from rag.generator import Generator
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"


def load_pipeline():
    index = Index.load(INDEX_PATH)
    indexer = Indexer()
    retriever = Retriever(index, indexer)
    generator = Generator()
    return retriever, generator


def ask_once(retriever, generator, question: str, mode: str) -> None:
    chunks = retriever.search(question, mode=mode)
    if not chunks:
        print("（未检索到相关内容）")
        return
    ans = generator.generate(question, chunks)
    print(f"\n问题：{question}")
    print(f"模式：{mode}")
    print(f"\n答案：\n{ans.answer}\n")
    print("引用来源：")
    for s in ans.sources:
        print(f"  [{s['index']}] 《{s['source']}》 {s['snippet'][:50]}...")


def interactive(retriever, generator, mode: str) -> None:
    print("制造设备故障知识库问答（输入 exit 退出）")
    while True:
        try:
            q = input(f"\n[{mode}] 提问 > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit", "q"}:
            break
        ask_once(retriever, generator, q, mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
    args = parser.parse_args()

    retriever, generator = load_pipeline()
    if args.question:
        ask_once(retriever, generator, args.question, args.mode)
    else:
        interactive(retriever, generator, args.mode)


if __name__ == "__main__":
    main()
