"""
title: DeepSeek Chat
author: OVINC CN
git_url: https://github.com/OVINC-CN/OpenWebUIPlugin.git
version: 0.1.0
licence: MIT
"""

import json
import logging
import time
import uuid
from typing import AsyncIterable, Literal, Optional, Tuple

import httpx
from fastapi import Request
from httpx import Response
from open_webui.env import GLOBAL_LOG_LEVEL
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(GLOBAL_LOG_LEVEL)


class APIException(Exception):
    def __init__(self, status: int, content: str, response: Response):
        self._status = status
        self._content = content
        self._response = response

    def __str__(self) -> str:
        # error msg
        try:
            return json.loads(self._content)["error"]["message"]
        except Exception:
            pass
        # build in error
        try:
            self._response.raise_for_status()
        except Exception as err:
            return str(err)
        return "Unknown API error"


class Pipe:
    class Valves(BaseModel):
        base_url: str = Field(default="https://api.deepseek.com", title="Base URL")
        api_key: str = Field(default="", title="API Key")
        allow_params: Optional[str] = Field(
            default="", title="透传参数", description="允许配置的参数，使用英文逗号分隔，例如 temperature"
        )
        timeout: int = Field(default=600, title="请求超时时间（秒）")
        proxy: Optional[str] = Field(default="", title="代理地址")
        models: str = Field(default="deepseek-v4-pro", title="模型", description="使用英文逗号分隔多个模型")

    class UserValves(BaseModel):
        thinking: bool = Field(default=True, title="思考模式")
        reasoning_effort: Literal["high", "max"] = Field(default="high", title="思考强度控制")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": model, "name": model} for model in self.valves.models.split(",") if model]

    async def pipe(self, body: dict, __user__: dict, __request__: Request) -> StreamingResponse:
        return StreamingResponse(self.__stream_pipe(body=body, __user__=__user__, __request__=__request__))

    async def __stream_pipe(self, body: dict, __user__: dict, __request__: Request) -> AsyncIterable:
        model, payload = await self._build_payload(body=body, user_valves=__user__["valves"])
        # call client
        async with httpx.AsyncClient(
            base_url=self.valves.base_url,
            headers={"Authorization": f"Bearer {self.valves.api_key}"},
            proxy=self.valves.proxy or None,
            trust_env=True,
            timeout=self.valves.timeout,
        ) as client:
            async with client.stream(**payload) as response:
                if response.status_code != 200:
                    text = ""
                    async for line in response.aiter_lines():
                        text += line  # pylint: disable=R1713
                    logger.error("response invalid with %d: %s", response.status_code, text)
                    raise APIException(status=response.status_code, content=text, response=response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("event:") or not line.startswith("data:"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if isinstance(line, str):
                        try:
                            line = json.loads(line)
                        except Exception:
                            continue
                    if not isinstance(line, dict):
                        continue
                    if line.get("usage"):
                        yield self._format_stream_data(model=model, usage=line.get("usage"))
                    choices = line.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    thinking_data = delta.get("reasoning_content") or ""
                    if thinking_data:
                        yield self._format_stream_data(model=model, reasoning_content=thinking_data)
                    content = delta.get("content") or ""
                    if content:
                        yield self._format_stream_data(model=model, content=content)

    async def _build_payload(self, body: dict, user_valves: UserValves, stream: bool = True) -> Tuple[str, dict]:
        model = body["model"].split(".", 1)[1]

        # build body
        data = {
            "model": model,
            "messages": body["messages"],
            "thinking": {
                "type": "enabled" if user_valves.thinking else "disabled",
            },
            "reasoning_effort": user_valves.reasoning_effort,
            "stream": stream,
            "stream_options": {
                "include_usage": True,
            },
        }

        # other parameters
        allowed_params = [k for k in self.valves.allow_params.split(",") if k]
        for key, val in body.items():
            if key in allowed_params:
                data[key] = val
        payload = {"method": "POST", "url": "/chat/completions", "json": data}

        return model, payload

    # pylint: disable=R0913,R0917
    def _format_stream_data(
        self,
        model: Optional[str] = "",
        content: Optional[str] = "",
        reasoning_content: Optional[str] = "",
        usage: Optional[dict] = None,
        if_finished: bool = False,
    ) -> str:
        data = {
            "id": f"chat.{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "choices": [],
            "created": int(time.time()),
            "model": model,
        }
        if content or reasoning_content:
            data["choices"] = [
                {
                    "finish_reason": "stop" if if_finished else "",
                    "index": 0,
                    "delta": {"content": content, "reasoning_content": reasoning_content},
                }
            ]
        if usage:
            data["usage"] = usage
        return f"data: {json.dumps(data)}\n\n"
