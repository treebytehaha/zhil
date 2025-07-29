# adapter/llm_adapter.py
import os
from openai import OpenAI

class LLMClient:
    def __init__(
        self,
        api_key: str | None   = None,
        base_url: str | None  = None,
        model: str | None     = None,
    ):
        self.api_key   = api_key  or os.getenv("DASHSCOPE_API_KEY")
        self.base_url  = base_url or os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model or os.getenv("DASHSCOPE_MODEL", "qwen-plus")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return resp.choices[0].message.content.strip()


