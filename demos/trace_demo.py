"""轻量 trace 可观测 demo：给多步调用加 span 层级 + 耗时统计。

解决简历里"trace 可观测标熟悉但只了解"的落差。核心讲清三件事：
  1. 为什么 Agent/RAG 要可观测：一次问答经过 检索→融合→重排→生成 多步，
     某一步变慢/失败要能定位到具体哪一步，而不是只看到"整次请求 3 秒"
  2. span 层级：父子关系对应调用栈，每步记录耗时 + 附加属性
  3. 自研 vs OTEL：自研就是"手写 span + 计时"（轻量、够用、无依赖）；
     OTEL 是标准化协议（trace_id/span_id 跨服务关联 + collector + 可视化后端），
     上生产要跨服务排查时才值得上。面试老实说"自研轻量 trace，没上 OTEL"。

用法：python demos/trace_demo.py
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Span:
    def __init__(self, name: str):
        self.name = name
        self.start = time.perf_counter()
        self.end = 0.0
        self.children: list["Span"] = []
        self.attrs: dict = {}

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000


class Tracer:
    """极简 span tracer：概念对齐 OTEL 的 span，实现自研，零依赖。"""

    def __init__(self):
        self.root: Span | None = None
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attrs):
        s = Span(name)
        s.attrs.update(attrs)
        if self._stack:
            self._stack[-1].children.append(s)
        else:
            self.root = s
        self._stack.append(s)
        try:
            yield s
        finally:
            s.end = time.perf_counter()
            self._stack.pop()

    def render(self) -> str:
        lines: list[str] = []

        def walk(span: Span, depth: int):
            attrs = " " + str(span.attrs) if span.attrs else ""
            lines.append(f"{'  ' * depth}{span.name:<14} {span.duration_ms:8.1f}ms{attrs}")
            for c in span.children:
                walk(c, depth + 1)

        if self.root:
            walk(self.root, 0)
        return "\n".join(lines)


tracer = Tracer()


def rag_ask(question: str) -> str:
    """模拟一次 RAG 问答，用 span 埋点每一步。真实场景换成 retriever/generator。"""
    with tracer.span("问答", question=question):
        with tracer.span("检索阶段"):
            with tracer.span("向量检索", top_k=20):
                time.sleep(0.08)  # 真实这里是 embedding + FAISS search
            with tracer.span("BM25 检索", top_k=20):
                time.sleep(0.02)
            with tracer.span("RRF 融合", k=60):
                time.sleep(0.01)
            with tracer.span("rerank 精排", top_k=5):
                time.sleep(0.06)
        with tracer.span("生成阶段"):
            time.sleep(0.35)  # 真实这里是 LLM 生成
            answer = "排查步骤：①核对锁模力设定值 ②清洁分型面 ③检查拉杆受力。"
    return answer


def main() -> None:
    question = "注塑机飞边怎么排查？"
    answer = rag_ask(question)
    print("trace 树（span 层级 + 每步耗时）：\n")
    print(tracer.render())
    print(f"\n最终答案：{answer}")
    print("\n说明：自研 trace = 手写 span + 计时，轻量够用；"
          "OTEL = 标准化协议 + trace_id 跨服务关联 + 可视化后端，上生产再上。")


if __name__ == "__main__":
    main()
