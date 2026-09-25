"""建库脚本：读取 data/docs 文档 → 切分 → embedding → FAISS + BM25 索引 → 保存。

用法：python build_index.py
"""
from __future__ import annotations

from config import DOCS_DIR, INDEX_DIR
from rag.indexer import Indexer
from rag.loader import load_documents

INDEX_PATH = INDEX_DIR / "index.pkl"


def main() -> None:
    chunks = load_documents(DOCS_DIR)
    print(f"读取文档，共切分 {len(chunks)} 个文本块")
    indexer = Indexer()
    index = indexer.build(chunks)
    index.save(INDEX_PATH)
    print(f"建库完成：{len(chunks)} 块 -> {INDEX_PATH}")


if __name__ == "__main__":
    main()
