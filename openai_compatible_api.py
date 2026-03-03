import argparse
import json
import time
import uuid
from typing import Any, Dict, Generator, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from infrastructure.llm_service import QwenService


class ChatMessage(BaseModel):
    role: str
    content: str | None = ""
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = Field(default=512, alias="max_completion_tokens")
    stream: bool = False
    tools: Optional[List[Dict[str, Any]]] = None

    class Config:
        populate_by_name = True
        extra = "allow"


def _build_response(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    content: str,
) -> Dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_chunks(model: str, stream_gen: Generator[str, None, None]):
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    first = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

    for piece in stream_gen:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    last = {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def create_app(llm: QwenService, served_model: str) -> FastAPI:
    app = FastAPI(title="AI Qwen OpenAI-Compatible API", version="0.1.0")

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": served_model,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ai-qwen-local",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if req.model != served_model:
            raise HTTPException(
                status_code=400,
                detail=f"model '{req.model}' 不可用，可用模型: '{served_model}'",
            )

        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        prompt_tokens = llm.get_token_count(str(messages))

        if req.stream:
            stream_gen = llm.generate_stream(
                messages=messages,
                tools=req.tools,
                max_new_tokens=req.max_tokens,
            )
            return StreamingResponse(
                _stream_chunks(served_model, stream_gen),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        content = llm.generate_response(
            messages=messages,
            tools=req.tools,
            max_new_tokens=req.max_tokens,
        )
        completion_tokens = llm.get_token_count(content)
        return JSONResponse(
            _build_response(
                model=served_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                content=content,
            )
        )

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="本地模型目录")
    parser.add_argument("--served-model-name", type=str, default="qwen-local")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--use-4bit", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()

    use_4bit = (
        args.use_4bit
        if args.use_4bit is not None
        else torch.cuda.is_available()
        and (torch.cuda.get_device_properties(0).total_memory / 1024**3 < 8)
    )

    llm = QwenService(args.model_path, use_4bit=use_4bit)
    app = create_app(llm, args.served_model_name)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
