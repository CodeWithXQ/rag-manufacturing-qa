"""Function Calling 最小 demo：演示工具调用的完整协议（OpenAI 兼容，DeepSeek）。

解决简历里"Function Calling 标熟悉但只了解"的落差。核心讲清三件事：
  1. 协议：LLM 不是只回文本，可以回 tool_calls（工具名 + 参数 JSON）
  2. 循环：你执行工具 → 结果塞回 messages（role=tool）→ 再让 LLM 总结
  3. 防御：参数校验（模型可能传错参数名/编造参数）、轮次限制（防死循环）、异常处理

用法：python demos/function_calling_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from config import get_llm_config

MAX_TURNS = 3  # 轮次限制：防止 LLM 反复调工具陷入死循环

# 工具定义（JSON Schema，OpenAI 兼容的 tools 格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_fault",
            "description": "查询制造设备故障的排查步骤",
            "parameters": {
                "type": "object",
                "properties": {
                    "device": {"type": "string", "description": "设备名，如 注塑机"},
                    "symptom": {"type": "string", "description": "故障现象，如 飞边"},
                },
                "required": ["device", "symptom"],
            },
        },
    }
]

# 模拟的故障知识库（真实场景这里是检索器 / 数据库 / 只读 API）
FAULT_DB = {
    "注塑机": {"飞边": "排查步骤：①核对锁模力设定值 ②清洁分型面 ③检查拉杆受力是否一致。"},
    "空压机": {"排气温度过高": "常见原因：①冷却器堵塞 ②油温过高 ③环境温度高 ④温控阀失效。"},
    "数控机床": {"主轴异响": "排查步骤：①检查轴承磨损 ②检查润滑 ③检查主轴动平衡。"},
}


def execute_tool(name: str, args: dict) -> str:
    """执行工具。真实场景：查库/调检索，且要做只读保护 + 防注入 + 越界检查。"""
    if name == "search_fault":
        return FAULT_DB.get(args.get("device", ""), {}).get(
            args.get("symptom", ""), "未找到该设备该故障的记录")
    return f"未知工具：{name}"


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    cfg = get_llm_config()
    url = f"{cfg['base_url'].rstrip('/')}/chat/completions"
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    if tools:
        payload["tools"] = tools
    resp = requests.post(url, json=payload, headers={
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }, timeout=cfg["timeout"])
    resp.raise_for_status()
    return resp.json()


def run_agent(query: str) -> str:
    messages = [{"role": "user", "content": query}]
    for turn in range(MAX_TURNS):
        data = chat(messages, TOOLS)
        msg = data["choices"][0]["message"]

        # 模型没要调工具 → 直接返回最终答案
        if not msg.get("tool_calls"):
            return msg.get("content", "")

        # 模型要调工具 → 逐个执行，结果回填
        messages.append(msg)  # 保留 assistant 消息（含 tool_calls）
        for tc in msg["tool_calls"]:
            fn = tc["function"]
            name = fn["name"]
            # 参数校验：arguments 是 JSON 字符串，模型可能传非法 JSON / 缺字段 / 编造
            try:
                args = json.loads(fn.get("arguments") or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, args)
            print(f"  [turn {turn + 1}] 调用工具 {name}({args})")
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
    return "达到最大轮次，未得到最终答案"


def main() -> None:
    queries = [
        "注塑机出现飞边怎么排查？",
        "帮我查一下空压机排气温度过高的原因",
    ]
    for q in queries:
        print("=" * 60)
        print(f"用户：{q}")
        answer = run_agent(q)
        print(f"最终答案：{answer}\n")


if __name__ == "__main__":
    main()
