"""文档读取 + 切分（chunking）。

支持 Markdown / TXT，按文档章节（## 标题）切分成独立块，并清理 Markdown 标记；
超过 chunk_size 的章节再按句子边界切分，相邻块保留 overlap。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP

_SENT_SPLIT = re.compile(r"(?<=[。！？；\n])")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str


def _clean_markdown(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^#+\s*", "", line)     # 去标题 # 符号
        line = re.sub(r"^[-*]\s+", "", line)   # 去列表符号
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences(sentences: list[str], chunk_size: int,
                     overlap: int) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if len(s) > chunk_size:
            if cur:
                chunks.append(cur)
                cur = ""
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(s), step):
                chunks.append(s[i:i + chunk_size])
            continue
        if len(cur) + len(s) <= chunk_size:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + s) if (overlap > 0 and cur) else s
    if cur:
        chunks.append(cur)
    return chunks


def split_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    sections = re.split(r"\n(?=##\s)", text)  # 按 ## 标题切章节
    # 文档主标题（# 开头）合并到第一个章节，避免产生纯标题低信息块
    if len(sections) >= 2 and sections[0].lstrip().startswith("# "):
        sections[1] = sections[0].strip() + "\n" + sections[1]
        sections = sections[1:]
    chunks: list[str] = []
    for sec in sections:
        cleaned = _clean_markdown(sec)
        if not cleaned:
            continue
        if len(cleaned) <= chunk_size:
            chunks.append(cleaned)
        else:
            chunks.extend(_merge_sentences(_split_sentences(cleaned),
                                           chunk_size, overlap))
    return chunks


def load_documents(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".markdown"}:
            continue
        text = path.read_text(encoding="utf-8")
        for i, chunk_text in enumerate(split_text(text)):
            chunks.append(Chunk(
                chunk_id=f"{path.name}#{i}",
                text=chunk_text,
                source=path.name,
            ))
    return chunks
