"""
title: OpenAI Responses
author: OVINC CN
git_url: https://github.com/OVINC-CN/OpenWebUIPlugin.git
version: 0.1.2
licence: MIT
"""

import base64
import binascii
import hashlib
import io
import json
import logging
import mimetypes
import re
import time
import uuid
from typing import AsyncIterable, List, Literal, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, Request, UploadFile
from httpx import Response
from open_webui.env import GLOBAL_LOG_LEVEL
from open_webui.models.users import UserModel, Users
from open_webui.routers.files import get_file_content_by_id, upload_file
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse

logger = logging.getLogger(__name__)
logger.setLevel(GLOBAL_LOG_LEVEL)

REASONING_EFFORT_MAP = {
    "关闭": "none",
    "低": "low",
    "中": "medium",
    "高": "high",
    "超高": "xhigh",
    "none": "none",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

SUMMARY_MAP = {
    "自动": "auto",
    "简要": "concise",
    "详尽": "detailed",
    "auto": "auto",
    "concise": "concise",
    "detailed": "detailed",
}

VERBOSITY_MAP = {
    "较短": "low",
    "适中": "medium",
    "较长": "high",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

IMAGE_MARKDOWN_PATTERN = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<url>[^)]+)\)")


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
        base_url: str = Field(default="https://api.openai.com/v1", title="Base URL")
        api_key: str = Field(default="", title="API Key")
        enable_reasoning: bool = Field(default=True, title="展示思考内容")
        allow_params: Optional[str] = Field(
            default="", title="透传参数", description="允许配置的参数，使用英文逗号分隔，例如 temperature"
        )
        timeout: int = Field(default=600, title="请求超时时间（秒）")
        proxy: Optional[str] = Field(default="", title="代理地址")
        models: str = Field(default="gpt-5", title="模型", description="使用英文逗号分隔多个模型")

    class UserValves(BaseModel):
        verbosity: Literal["较短", "适中", "较长"] = Field(default="适中", title="输出详细程度")
        reasoning_effort: Literal["关闭", "低", "中", "高", "超高"] = Field(
            default="中", title="推理强度"
        )
        summary: Literal["自动", "简要", "详尽"] = Field(default="自动", title="思考输出摘要程度")

    def __init__(self):
        self.valves = self.Valves()

    def pipes(self):
        return [{"id": model, "name": model} for model in self.valves.models.split(",") if model]

    async def pipe(self, body: dict, __user__: dict, __request__: Request) -> StreamingResponse:
        return StreamingResponse(self.__stream_pipe(body=body, __user__=__user__, __request__=__request__))

    async def __stream_pipe(self, body: dict, __user__: dict, __request__: Request) -> AsyncIterable:
        user = Users.get_user_by_id(__user__["id"])
        if not user:
            raise ValueError("user not found")
        model, payload = await self._build_payload(user=user, body=body, user_valves=__user__["valves"])
        emitted_image_call_ids = set()
        emitted_image_hashes = set()
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
                is_thinking = self.valves.enable_reasoning
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("event:") or not line.startswith("data:"):
                        continue
                    if line.startswith("data: "):
                        line = line[6:]
                    if isinstance(line, str):
                        line = json.loads(line)
                    match line.get("type"):
                        case "response.reasoning_summary_text.delta":
                            if is_thinking:
                                yield self._format_stream_data(model=model, reasoning_content=line["delta"])
                        case "response.output_text.delta":
                            if is_thinking:
                                is_thinking = False
                            yield self._format_stream_data(model=model, content=line["delta"])
                        case "response.output_item.done":
                            item = line.get("item") or {}
                            if item.get("type") == "image_generation_call":
                                image_content = self._format_image_generation_call(
                                    __request__=__request__,
                                    user=user,
                                    item=item,
                                    payload=payload["json"],
                                    emitted_image_call_ids=emitted_image_call_ids,
                                    emitted_image_hashes=emitted_image_hashes,
                                )
                                if image_content:
                                    yield self._format_stream_data(model=model, content=image_content)
                        case "response.completed":
                            for image_content in self._format_completed_image_generation_calls(
                                __request__=__request__,
                                user=user,
                                response=line.get("response") or {},
                                payload=payload["json"],
                                emitted_image_call_ids=emitted_image_call_ids,
                                emitted_image_hashes=emitted_image_hashes,
                            ):
                                yield self._format_stream_data(model=model, content=image_content)
                            yield self._format_stream_data(
                                model=model, usage=line["response"]["usage"], if_finished=True
                            )
                        case _:
                            event_type = line["type"]
                            if event_type.endswith("in_progress") or event_type.endswith("completed"):
                                event_type_split = event_type.split(".")[1:]
                                if len(event_type_split) == 2:
                                    data = {
                                        "event": {
                                            "type": "status",
                                            "data": {
                                                "description": " ".join(event_type_split),
                                                "done": event_type_split[1] == "completed",
                                            },
                                        }
                                    }
                                    yield f"data: {json.dumps(data)}\n\n"

    async def _build_payload(
        self,
        user: UserModel,
        body: dict,
        user_valves: UserValves,
        stream: bool = True,
    ) -> Tuple[str, dict]:
        model = body["model"].split(".", 1)[1]

        # build messages
        messages = []
        for message in body["messages"]:
            role = message["role"]
            message_content = message["content"]
            if role == "assistant":
                assistant_content, reference_images = await self._parse_assistant_message(
                    user=user,
                    message_content=message_content,
                )
                if assistant_content:
                    messages.append({"content": assistant_content, "role": role})
                if reference_images:
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "参考上一条助理回复中的生成图片。"},
                                *reference_images,
                            ],
                        }
                    )
                continue

            if isinstance(message_content, str):
                messages.append(
                    {
                        "content": (
                            await self._parse_message_text(user=user, text=message_content)
                            if role == "user"
                            else message_content
                        ),
                        "role": role,
                    }
                )
            elif isinstance(message_content, list):
                content = []
                for item in message_content:
                    if item["type"] == "text":
                        content.extend(
                            await self._parse_message_text_as_content(user=user, text=item["text"])
                            if role == "user"
                            else [{"type": "input_text", "text": item["text"]}]
                        )
                    elif item["type"] in {"input_text", "output_text"}:
                        content.extend(
                            await self._parse_message_text_as_content(user=user, text=item["text"])
                            if role == "user"
                            else [{"type": "input_text", "text": item["text"]}]
                        )
                    elif item["type"] in {"image_url", "input_image"}:
                        if role != "user":
                            continue
                        image_content = self._normalize_input_image_item(item)
                        if not image_content:
                            raise TypeError("Invalid image content")
                        content.append(image_content)
                    else:
                        raise TypeError("Invalid message content type %s" % item["type"])
                messages.append({"role": role, "content": content})
            else:
                raise TypeError("Invalid message content type %s" % type(message_content))

        # reasoning
        reasoning_effort = REASONING_EFFORT_MAP[user_valves.reasoning_effort]

        # build body
        data = {
            "model": model,
            "input": messages,
            "reasoning": {
                "effort": reasoning_effort,
                "summary": SUMMARY_MAP[user_valves.summary],
            },
            "text": {
                "verbosity": VERBOSITY_MAP[user_valves.verbosity],
            },
            "stream": stream,
            "store": False,
        }

        # max tokens
        if "max_completion_tokens" in body:
            data["max_output_tokens"] = body["max_completion_tokens"]
        elif "max_tokens" in body:
            data["max_output_tokens"] = body["max_tokens"]

        # other parameters
        allowed_params = [k for k in self.valves.allow_params.split(",") if k]
        for key, val in body.items():
            if key in allowed_params:
                data[key] = val
        payload = {"method": "POST", "url": "/responses", "json": data}

        # check tools
        if body.get("tools", []):
            payload["json"]["tools"] = body["tools"]

        return model, payload

    async def _parse_assistant_message(self, user: UserModel, message_content) -> Tuple[object, List[dict]]:
        if isinstance(message_content, str):
            return message_content, await self._extract_markdown_images(user=user, text=message_content)

        if not isinstance(message_content, list):
            raise TypeError("Invalid message content type %s" % type(message_content))

        assistant_content = []
        reference_images = []
        for item in message_content:
            item_type = item.get("type")
            if item_type == "refusal":
                assistant_content.append({"type": "refusal", "refusal": item.get("refusal", "")})
                continue
            if item_type in {"text", "input_text", "output_text"}:
                text = item.get("text", "")
                assistant_content.append({"type": "output_text", "text": text})
                reference_images.extend(await self._extract_markdown_images(user=user, text=text))
                continue
            if item_type in {"image_url", "input_image"}:
                image_content = self._normalize_input_image_item(item)
                if image_content:
                    reference_images.append(image_content)
                continue
            raise TypeError("Invalid message content type %s" % item_type)

        if len(assistant_content) == 1 and assistant_content[0].get("type") == "output_text":
            return assistant_content[0].get("text", ""), reference_images
        return assistant_content, reference_images

    async def _parse_message_text(self, user: UserModel, text: str):
        content = await self._parse_message_text_as_content(user=user, text=text)
        if len(content) == 1 and content[0].get("type") == "input_text" and content[0].get("text") == text:
            return text
        return content

    async def _parse_message_text_as_content(self, user: UserModel, text: str) -> List[dict]:
        content = []
        cursor = 0
        has_image = False

        for match in IMAGE_MARKDOWN_PATTERN.finditer(text):
            before = text[cursor : match.start()]
            if before.strip():
                content.append({"type": "input_text", "text": before})

            image_content = await self._parse_markdown_image(
                user=user,
                alt_text=match.group("alt"),
                image_url=match.group("url"),
            )
            if image_content:
                has_image = True
                content.append(image_content)
            else:
                content.append({"type": "input_text", "text": match.group(0)})

            cursor = match.end()

        remaining = text[cursor:]
        if remaining.strip():
            content.append({"type": "input_text", "text": remaining})

        if not has_image:
            return [{"type": "input_text", "text": text}]
        return content or [{"type": "input_text", "text": text}]

    async def _extract_markdown_images(self, user: UserModel, text: str) -> List[dict]:
        images = []
        seen = set()
        for match in IMAGE_MARKDOWN_PATTERN.finditer(text):
            image_content = await self._parse_markdown_image(
                user=user,
                alt_text=match.group("alt"),
                image_url=match.group("url"),
            )
            if not image_content:
                continue
            image_key = image_content.get("file_id") or image_content.get("image_url")
            if image_key in seen:
                continue
            seen.add(image_key)
            images.append(image_content)
        return images

    async def _parse_markdown_image(self, user: UserModel, alt_text: str, image_url: str) -> Optional[dict]:
        image_url = image_url.strip()
        if image_url.startswith("<") and image_url.endswith(">"):
            image_url = image_url[1:-1].strip()

        if image_url.startswith(("http://", "https://", "data:")):
            return {"type": "input_image", "image_url": image_url}

        file_id = self._extract_file_id_from_markdown(alt_text=alt_text, image_url=image_url)
        if not file_id:
            return None

        data_url = await self._get_image_data_url_from_file(user=user, file_id=file_id)
        if not data_url:
            return None
        return {"type": "input_image", "image_url": data_url}

    async def _get_image_data_url_from_file(self, user: UserModel, file_id: str) -> str:
        try:
            file_response = await get_file_content_by_id(id=file_id, user=user)
            with open(file_response.path, "rb") as file_content:
                image_bytes = file_content.read()
        except Exception as err:
            logger.warning("failed to load generated image %s: %s", file_id, err)
            return ""

        mime_type = mimetypes.guess_type(file_response.path)[0] or "image/png"
        encoded = base64.b64encode(image_bytes).decode()
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_input_image_item(item: dict) -> dict:
        image_content = {"type": "input_image"}
        image_url = item.get("image_url", "")
        if isinstance(image_url, dict):
            image_url = image_url.get("url", "")
        if isinstance(image_url, str) and image_url:
            image_content["image_url"] = image_url
        elif item.get("file_id"):
            image_content["file_id"] = item["file_id"]
        else:
            return {}

        if item.get("detail"):
            image_content["detail"] = item["detail"]
        return image_content

    @staticmethod
    def _extract_file_id_from_markdown(alt_text: str, image_url: str) -> str:
        if alt_text.startswith("openai-image-partial-"):
            return alt_text.removeprefix("openai-image-partial-")
        if alt_text.startswith("openai-image-"):
            return alt_text.removeprefix("openai-image-")

        match = re.search(r"/files/([^/?#]+)/", image_url)
        if match:
            return match.group(1)
        match = re.search(r"/files/([^/?#]+)", image_url)
        if match:
            return match.group(1)
        return ""

    def _format_completed_image_generation_calls(
        self,
        __request__: Request,
        user: UserModel,
        response: dict,
        payload: dict,
        emitted_image_call_ids: set,
        emitted_image_hashes: set,
    ):
        for item in response.get("output", []):
            if item.get("type") != "image_generation_call":
                continue
            image_content = self._format_image_generation_call(
                __request__=__request__,
                user=user,
                item=item,
                payload=payload,
                emitted_image_call_ids=emitted_image_call_ids,
                emitted_image_hashes=emitted_image_hashes,
            )
            if image_content:
                yield image_content

    def _format_image_generation_call(
        self,
        __request__: Request,
        user: UserModel,
        item: dict,
        payload: dict,
        emitted_image_call_ids: set,
        emitted_image_hashes: set,
    ) -> str:
        image_call_id = item.get("id")
        if image_call_id and image_call_id in emitted_image_call_ids:
            return ""

        image_content = self._format_image_generation_result(
            __request__=__request__,
            user=user,
            image_data=item.get("result", ""),
            mime_type=self._get_image_tool_mime_type(payload, item),
            image_prefix="openai-image",
            emitted_image_hashes=emitted_image_hashes,
        )
        if image_content and image_call_id:
            emitted_image_call_ids.add(image_call_id)
        return image_content

    def _format_image_generation_result(
        self,
        __request__: Request,
        user: UserModel,
        image_data: str,
        mime_type: str,
        image_prefix: str,
        emitted_image_hashes: set,
    ) -> str:
        if not image_data:
            return ""
        image_bytes = self._decode_base64_image(image_data)
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash in emitted_image_hashes:
            return ""

        file_item = upload_file(
            request=__request__,
            background_tasks=BackgroundTasks(),
            file=UploadFile(
                file=io.BytesIO(image_bytes),
                filename=f"generated-image-{uuid.uuid4().hex}{self._get_image_extension(mime_type)}",
                headers=Headers({"content-type": mime_type}),
            ),
            process=False,
            user=user,
            metadata={"mime_type": mime_type},
        )
        emitted_image_hashes.add(image_hash)
        image_url = __request__.app.url_path_for("get_file_content_by_id", id=file_item.id)
        return f"![{image_prefix}-{file_item.id}]({image_url})"

    @staticmethod
    def _decode_base64_image(image_data: str) -> bytes:
        data = image_data.strip()
        if data.startswith("data:") and "," in data:
            data = data.split(",", 1)[1]

        data = "".join(data.split())
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            padding = len(data) % 4
            if padding:
                data = f"{data}{'=' * (4 - padding)}"
            decoded = base64.b64decode(data)

        if not decoded:
            raise ValueError("decoded image bytes is empty")
        return decoded

    @staticmethod
    def _get_image_tool_mime_type(payload: dict, item: dict) -> str:
        output_format = item.get("output_format")
        if not output_format:
            for tool in payload.get("tools", []):
                if tool.get("type") == "image_generation":
                    output_format = tool.get("output_format")
                    break

        return {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(output_format or "", "image/png")

    @staticmethod
    def _get_image_extension(mime_type: str) -> str:
        file_ext = mimetypes.guess_extension(mime_type) or ".png"
        return ".jpg" if file_ext == ".jpe" else file_ext

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
