"""核心纯逻辑单元测试（不依赖模型权重，可离线运行）。"""
from __future__ import annotations

from rag.generator import build_prompt, extract_sources
from rag.indexer import tokenize
from rag.loader import Chunk, _clean_markdown, split_text
from rag.retriever import Retriever


def test_clean_markdown():
    text = "# 标题\n## 一、现象\n- 列表项\n正常文本"
    cleaned = _clean_markdown(text)
    assert "#" not in cleaned
    assert "列表项" in cleaned
    assert "标题" in cleaned


def test_split_text_by_section():
    text = "# 故障A\n\n## 一、现象\n现象描述。\n\n## 二、原因\n原因描述。"
    chunks = split_text(text)
    # 主标题合并到第一个章节，共 2 个块
    assert len(chunks) == 2
    assert "故障A" in chunks[0]
    assert "现象" in chunks[0]
    assert "原因" in chunks[1]


def test_tokenize():
    tokens = tokenize("注塑机飞边怎么排查")
    assert len(tokens) > 0
    assert all(t.strip() for t in tokens)


def test_rrf_fusion_dedup_and_rank():
    a = Chunk("a", "ta", "d1")
    b = Chunk("b", "tb", "d2")
    c = Chunk("c", "tc", "d3")
    v_res = [(a, 0.9), (b, 0.8)]
    b_res = [(c, 10.0), (a, 5.0)]
    merged = Retriever._rrf_fusion(v_res, b_res, k=60)
    # a 同时出现在两路，RRF 分最高，应排第一；去重后共 3 个
    assert merged[0][0].chunk_id == "a"
    assert len(merged) == 3


def test_build_prompt_contains_refs():
    chunks = [(Chunk("a", "原文内容A", "01.md"), 0.9),
              (Chunk("b", "原文内容B", "02.md"), 0.8)]
    prompt = build_prompt("问题", chunks)
    assert "[1]" in prompt and "[2]" in prompt
    assert "01.md" in prompt and "原文内容A" in prompt


def test_extract_sources():
    chunks = [(Chunk("a", "x" * 300, "01.md"), 0.9)]
    sources = extract_sources(chunks)
    assert sources[0]["source"] == "01.md"
    assert sources[0]["index"] == 1
    assert len(sources[0]["snippet"]) <= 200
