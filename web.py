"""Gradio Web 界面。用法：python web.py，然后浏览器打开 http://127.0.0.1:7860。"""
from __future__ import annotations

import gradio as gr

from config import INDEX_DIR
from rag.generator import Generator
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"

index = Index.load(INDEX_PATH)
indexer = Indexer()
retriever = Retriever(index, indexer)
generator = Generator()


def answer(question: str, mode: str):
    if not question.strip():
        return "请输入问题", ""
    chunks = retriever.search(question, mode=mode)
    if not chunks:
        return "（未检索到相关内容）", ""
    ans = generator.generate(question, chunks)
    sources_text = "\n".join(
        f"[{s['index']}] 《{s['source']}》\n   {s['snippet']}" for s in ans.sources)
    return ans.answer, sources_text


with gr.Blocks(title="制造设备故障知识库问答 Agent") as demo:
    gr.Markdown("# 制造设备故障知识库问答 Agent（RAG）")
    gr.Markdown("**核心亮点：自研混合检索**（向量检索 + BM25 关键词检索 + RRF 融合 + rerank 精排）")
    with gr.Row():
        mode = gr.Radio(["hybrid", "vector"], value="hybrid",
                        label="检索模式（hybrid=混合检索，vector=纯向量 baseline）")
    question = gr.Textbox(label="提问", placeholder="例如：注塑机飞边怎么排查处理？")
    btn = gr.Button("提问", variant="primary")
    answer_box = gr.Textbox(label="答案", lines=10)
    sources_box = gr.Textbox(label="引用来源", lines=6)
    btn.click(answer, inputs=[question, mode],
              outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
