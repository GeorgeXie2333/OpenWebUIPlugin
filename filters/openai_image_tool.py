"""
title: OpenAI Image Tool
author: OVINC CN
git_url: https://github.com/OVINC-CN/OpenWebUIPlugin.git
description: OpenAI Image Generation Tool
version: 0.0.1
licence: MIT
"""

from typing import Literal

from pydantic import BaseModel, Field

QUALITY_MAP = {
    "低": "low",
    "中": "medium",
    "高": "high",
    "自动": "auto",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "auto": "auto",
}


class Filter:
    class Valves(BaseModel):
        priority: int = Field(default=0, description="filter priority")
        image_model: str = Field(
            default="gpt-image-2",
            title="图片生成模型",
            description="留空时使用 OpenAI 默认图像生成模型",
        )
        quality: Literal["低", "中", "高", "自动"] = Field(default="自动", title="图片质量")

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True
        self.icon = (
            "data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgNDggNDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3Ln"
            "czLm9yZy8yMDAwL3N2ZyIgc3Ryb2tlPSJjdXJyZW50Q29sb3IiIHN0cm9rZS13aWR0aD0iNCIgc3Ryb2tlLWxpbmVjYXA9ImJ1dHQiIH"
            "N0cm9rZS1saW5lam9pbj0ibWl0ZXIiPjxyZWN0IHg9IjYiIHk9IjgiIHdpZHRoPSIzNiIgaGVpZ2h0PSIzMiIgcng9IjIiPjwvcmVjdD"
            "48cGF0aCBkPSJNMTIgMzNsOC05IDcgNyA1LTYgNCA4Ij48L3BhdGg+PGNpcmNsZSBjeD0iMzEiIGN5PSIxNyIgcj0iNCI+PC9jaXJjbG"
            "U+PC9zdmc+"
        )

    def inlet(self, body: dict) -> dict:
        tool = {"type": "image_generation"}
        image_model = self.valves.image_model.strip()
        if image_model:
            tool["model"] = image_model
        tool["quality"] = QUALITY_MAP[self.valves.quality]

        if body.get("tools"):
            body["tools"].append(tool)
        else:
            body["tools"] = [tool]
        return body
