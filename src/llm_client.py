"""
Mini-MA-RAG: 统一 LLM 客户端

封装 OpenAI 兼容 API 的调用，支持普通生成和结构化输出。
"""

import json
import os
from typing import Type, Optional, Any
from pydantic import BaseModel
from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url or OPENAI_BASE_URL
        self.model = model or MODEL_NAME
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        system_message: str,
        user_message: str,
        temperature: Optional[float] = None,
    ) -> str:
        """普通对话生成，返回字符串"""
        temp = temperature if temperature is not None else self.temperature
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temp,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"LLM API 调用失败（已重试{self.max_retries}次）: {e}")
        return ""

    def chat_structured(
        self,
        system_message: str,
        user_message: str,
        output_schema: Type[BaseModel],
        temperature: Optional[float] = None,
    ) -> BaseModel:
        """结构化输出，返回 Pydantic 对象"""
        temp = temperature if temperature is not None else self.temperature
        for attempt in range(self.max_retries):
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=output_schema,
                    temperature=temp,
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                if attempt == self.max_retries - 1:
                    # Fallback: try json_mode if parse fails
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": system_message},
                                {"role": "user", "content": user_message},
                            ],
                            response_format={"type": "json_object"},
                            temperature=temp,
                        )
                        content = response.choices[0].message.content
                        data = json.loads(content)
                        return output_schema(**data)
                    except Exception as e2:
                        raise RuntimeError(
                            f"结构化输出失败（已重试{self.max_retries}次）。"
                            f"原错误: {e}; Fallback错误: {e2}"
                        )
        return output_schema()  # type: ignore


# 全局默认客户端实例
_default_client: Optional[LLMClient] = None


def get_default_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
