"""端到端生成评测：引用命中率（规则）+ 幻觉率（LLM-as-judge）。

链路：问题 → 混合检索 top-5 → LLM 生成（答案带 [i] 引用编号）→ 评测

指标（回答"检索好了，生成质量到底如何"）：
  1. 引用覆盖率 citation_rate：答案是否标注了引用编号（没标 = 潜在幻觉信号）
  2. 引用命中率 citation_hit ：答案引用的文档中，是否命中标准答案文档 gold_doc
  3. 幻觉率   hallucination ：LLM-as-judge 判断答案是否编造/曲解了检索到的上下文

用法：
  python evaluate_generation.py              # 全量 50 题（约 100 次 LLM 调用）
  python evaluate_generation.py --limit 2    # 小样本验证
  python evaluate_generation.py --no-judge   # 只跑引用指标，跳过 judge（省一半调用）
"""
from __future__ import annotations

import argparse
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

from config import FINAL_TOP_K, INDEX_DIR, TESTSET_DIR
from rag.generator import Generator
from rag.indexer import Index, Indexer
from rag.retriever import Retriever

INDEX_PATH = INDEX_DIR / "index.pkl"

CITED_RE = re.compile(r"\[(\d+)\]")

JUDGE_SYSTEM = (
    "你是一名严格的 RAG 答案质量评审员，负责判断模型答案是否存在「幻觉」。"
    "幻觉指：答案陈述了参考资料中不存在、或与参考资料相矛盾的内容，"
    "尤其是故障原因、具体数值、操作步骤、专业诊断结论。"
)


def load_testset() -> list[dict]:
    return json.loads(
        (TESTSET_DIR / "testset.json").read_text(encoding="utf-8"))


def extract_cited_indices(answer: str) -> list[int]:
    return [int(x) for x in CITED_RE.findall(answer)]


def cited_sources(answer: str, chunks) -> list[str]:
    """答案引用的编号映射回 chunk 来源文档（编号越界或非法则忽略）。"""
    srcs = []
    n = len(chunks)
    for i in extract_cited_indices(answer):
        if 1 <= i <= n:
            srcs.append(chunks[i - 1][0].source)
    return srcs


def build_judge_prompt(question: str, chunks, answer: str) -> str:
    refs = "\n\n".join(
        f"[{i}] 出处《{c.source}》\n{c.text}"
        for i, (c, _) in enumerate(chunks, start=1))
    return (
        f"用户问题：{question}\n\n"
        f"参考资料：\n{refs}\n\n"
        f"模型答案：\n{answer}\n\n"
        f"请判断模型答案是否存在幻觉（编造或曲解参考资料）。"
        f"只输出 JSON，不要输出其他内容：{{\"hallucinated\": true 或 false}}"
    )


def parse_judge(text: str) -> bool | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return bool(json.loads(m.group(0)).get("hallucinated"))
    except (json.JSONDecodeError, AttributeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 题")
    ap.add_argument("--no-judge", action="store_true", help="跳过幻觉 judge")
    args = ap.parse_args()

    index = Index.load(INDEX_PATH)
    indexer = Indexer()
    retriever = Retriever(index, indexer)
    gen = Generator()

    testset = load_testset()
    if args.limit:
        testset = testset[:args.limit]

    n = len(testset)
    n_cited = 0          # 答案标注了引用的题数
    n_hit = 0            # 引用命中 gold_doc 的题数
    n_hallucinated = 0   # judge 判定有幻觉的题数
    n_judge_failed = 0   # judge 输出无法解析的题数
    n_gen_failed = 0     # 生成失败的题数

    miss_hit = []        # 引用未命中 gold_doc 的题号
    hallucinated = []    # 判为幻觉的题号
    no_cite = []         # 无引用的题号

    for item in testset:
        q = item["question"]
        gold = item["gold_doc"]
        try:
            results = retriever.hybrid_search(q, final_top_k=FINAL_TOP_K)
            answer = gen.generate(q, results).answer
        except Exception as e:  # 单题失败不中断整体评测
            n_gen_failed += 1
            print(f"[{item['id']}] 生成失败：{e}")
            continue

        srcs = cited_sources(answer, results)
        if srcs:
            n_cited += 1
            if gold in srcs:
                n_hit += 1
            else:
                miss_hit.append(item["id"])
        else:
            no_cite.append(item["id"])

        if not args.no_judge:
            judge_prompt = build_judge_prompt(q, results, answer)
            try:
                raw = gen.llm.chat(
                    [{"role": "system", "content": JUDGE_SYSTEM},
                     {"role": "user", "content": judge_prompt}],
                    temperature=0.0, max_tokens=64)
                verdict = parse_judge(raw)
                if verdict is None:
                    n_judge_failed += 1
                elif verdict:
                    n_hallucinated += 1
                    hallucinated.append(item["id"])
            except Exception as e:
                n_judge_failed += 1
                print(f"[{item['id']}] judge 调用失败：{e}")

    total = n - n_gen_failed
    print(f"端到端生成评测（混合检索 top{FINAL_TOP_K} → LLM 生成）")
    print(f"评测题数：{n}（生成失败 {n_gen_failed}，有效 {total}）\n")
    print(f"{'指标':<16}{'数值':>10}")
    print("-" * 26)
    print(f"{'引用覆盖率':<16}{n_cited / total:>10.2%}")
    print(f"{'引用命中率':<16}{n_hit / total:>10.2%}")
    if not args.no_judge:
        print(f"{'幻觉率':<16}{n_hallucinated / total:>10.2%}"
              f"  (judge 无法解析 {n_judge_failed})")
    print()
    print(f"引用未命中 gold_doc：{miss_hit}")
    print(f"未标注引用：{no_cite}")
    if not args.no_judge:
        print(f"判定幻觉题号：{hallucinated}")


if __name__ == "__main__":
    main()
