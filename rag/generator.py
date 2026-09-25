"""生成模块：拼接 prompt、调用 LLM、解析引用来源。"""
from __future__ import annotations

from dataclasses import dataclass

from rag.llm import LLMClient
from rag.loader import Chunk

SYSTEM_PROMPT = "你是一名制造设备故障诊断专家，请严格依据提供的参考资料回答问题。"


@dataclass
class Answer:
    answer: str
    sources: list[dict]  # [{"index": 1, "source": "xx.md", "snippet": "..."}]


def build_prompt(question: str, chunks: list[tuple[Chunk, float]]) -> str:
    refs = []
    for i, (chunk, _) in enumerate(chunks, start=1):
        refs.append(f"[{i}] 出处《{chunk.source}》\n{chunk.text}")
    context = "\n\n".join(refs)
    return (
        f"参考资料：\n{context}\n\n"
        f"请根据以上参考资料回答用户问题，并在答案中用 [1]、[2] 等编号标注引用来源。\n"
        f"若资料不足以回答，请明确说明。\n\n"
        f"用户问题：{question}"
    )


def extract_sources(chunks: list[tuple[Chunk, float]]) -> list[dict]:
    return [
        {"index": i, "source": c.source, "snippet": c.text[:200]}
        for i, (c, _) in enumerate(chunks, start=1)
    ]


class Generator:
    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def generate(self, question: str,
                 chunks: list[tuple[Chunk, float]]) -> Answer:
        prompt = build_prompt(question, chunks)
        answer_text = self.llm.generate(prompt, system=SYSTEM_PROMPT)
        return Answer(answer=answer_text, sources=extract_sources(chunks))
