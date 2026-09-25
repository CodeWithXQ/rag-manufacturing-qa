"""统一 LLM 接入层：OpenAI 兼容协议，requests 直连，不依赖 openai SDK。

DeepSeek / 通义千问 / 智谱均提供 OpenAI 兼容接口，只需在 .env 切换
base_url + api_key + model 即可，核心调用逻辑自己实现，便于讲清原理。
"""
from __future__ import annotations

import requests

from config import get_llm_config


class LLMClient:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or get_llm_config()
        self.base_url = self.cfg["base_url"].rstrip("/")

    def chat(self, messages: list[dict], temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.cfg['api_key']}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": self.cfg["temperature"] if temperature is None else temperature,
            "max_tokens": self.cfg["max_tokens"] if max_tokens is None else max_tokens,
            "stream": False,
        }
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=self.cfg["timeout"])
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)
