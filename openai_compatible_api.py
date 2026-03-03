import argparse
import gzip
import json
import re
import time
import uuid
import zlib
from typing import Any, Dict, Generator, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from infrastructure.llm_service import QwenService

try:
    from compression import zstd as _zstd_builtin  # Python 3.14+
except Exception:
    _zstd_builtin = None

try:
    import zstandard as _zstd_ext  # optional fallback
except Exception:
    _zstd_ext = None


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

    model_config = ConfigDict(populate_by_name=True, extra="allow")


class ResponsesRequest(BaseModel):
    model: str
    input: Any
    stream: bool = False
    temperature: float = 0.7
    max_output_tokens: int = 512
    tools: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


def _extract_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    if not text:
        return calls

    blocks = re.findall(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    blocks += re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL | re.IGNORECASE)

    for block in blocks:
        try:
            obj = json.loads(block.strip())
            if isinstance(obj, dict) and obj.get("name"):
                calls.append({"name": str(obj["name"]), "arguments": obj.get("arguments", {})})
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and item.get("name"):
                        calls.append({"name": str(item["name"]), "arguments": item.get("arguments", {})})
        except Exception:
            continue
    return calls


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


def _responses_input_to_messages(raw_input: Any) -> List[Dict[str, str]]:
    if isinstance(raw_input, str):
        return [{"role": "user", "content": raw_input}]

    if isinstance(raw_input, list):
        messages: List[Dict[str, str]] = []
        for item in raw_input:
            if isinstance(item, dict):
                item_type = str(item.get("type", ""))
                if item_type == "function_call_output":
                    call_id = str(item.get("call_id", "tool"))
                    output = item.get("output", "")
                    messages.append(
                        {
                            "role": "tool",
                            "name": call_id,
                            "content": str(output),
                        }
                    )
                    continue
                if item_type == "function_call":
                    # Assistant-side planning item; no need to feed back as user text.
                    continue

                role = str(item.get("role", "user"))
                content = item.get("content", "")
                if isinstance(content, list):
                    # responses API content can be an array of typed blocks.
                    text_parts = []
                    for c in content:
                        if isinstance(c, dict):
                            if c.get("type") in {"input_text", "output_text", "text"}:
                                text_parts.append(str(c.get("text", "")))
                            elif c.get("type") == "input_image":
                                text_parts.append("[image]")
                            elif "content" in c:
                                text_parts.append(str(c.get("content", "")))
                    content = "\n".join(p for p in text_parts if p)
                messages.append({"role": role, "content": str(content)})
        if messages:
            return messages

    return [{"role": "user", "content": str(raw_input)}]



def _build_responses_json(
    *,
    model: str,
    content: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    output_id = f"msg_{uuid.uuid4().hex[:24]}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": output_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": content,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _build_responses_tool_json(
    *,
    model: str,
    tool_calls: List[Dict[str, Any]],
    prompt_tokens: int,
    completion_tokens: int,
) -> Dict[str, Any]:
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    output = []
    for call in tool_calls:
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "status": "completed",
                "call_id": f"call_{uuid.uuid4().hex[:24]}",
                "name": call["name"],
                "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
            }
        )
    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": output,
        "output_text": "",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _stream_responses_chunks(model: str, stream_gen: Generator[str, None, None]):
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    item_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
    }

    created_evt = {"type": "response.created", "response": {**base_response, "status": "in_progress"}}
    yield f"event: response.created\ndata: {json.dumps(created_evt, ensure_ascii=False)}\n\n"

    in_progress_evt = {"type": "response.in_progress", "response": {**base_response, "status": "in_progress"}}
    yield f"event: response.in_progress\ndata: {json.dumps(in_progress_evt, ensure_ascii=False)}\n\n"

    item_added_evt = {
        "type": "response.output_item.added",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        },
    }
    yield f"event: response.output_item.added\ndata: {json.dumps(item_added_evt, ensure_ascii=False)}\n\n"

    part_added_evt = {
        "type": "response.content_part.added",
        "response_id": response_id,
        "output_index": 0,
        "item_id": item_id,
        "content_index": 0,
        "part": {"type": "output_text", "text": "", "annotations": []},
    }
    yield f"event: response.content_part.added\ndata: {json.dumps(part_added_evt, ensure_ascii=False)}\n\n"

    chunks: List[str] = []
    for piece in stream_gen:
        chunks.append(piece)
        delta_evt = {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "output_index": 0,
            "item_id": item_id,
            "content_index": 0,
            "delta": piece,
        }
        yield f"event: response.output_text.delta\ndata: {json.dumps(delta_evt, ensure_ascii=False)}\n\n"

    final_text = "".join(chunks)
    text_done_evt = {
        "type": "response.output_text.done",
        "response_id": response_id,
        "output_index": 0,
        "item_id": item_id,
        "content_index": 0,
        "text": final_text,
    }
    yield f"event: response.output_text.done\ndata: {json.dumps(text_done_evt, ensure_ascii=False)}\n\n"

    part_done_evt = {
        "type": "response.content_part.done",
        "response_id": response_id,
        "output_index": 0,
        "item_id": item_id,
        "content_index": 0,
        "part": {"type": "output_text", "text": final_text, "annotations": []},
    }
    yield f"event: response.content_part.done\ndata: {json.dumps(part_done_evt, ensure_ascii=False)}\n\n"

    item_done_evt = {
        "type": "response.output_item.done",
        "response_id": response_id,
        "output_index": 0,
        "item": {
            "id": item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": final_text, "annotations": []}],
        },
    }
    yield f"event: response.output_item.done\ndata: {json.dumps(item_done_evt, ensure_ascii=False)}\n\n"

    completed_evt = {
        "type": "response.completed",
        "response": {
            **base_response,
            "status": "completed",
            "output": [
                {
                    "id": item_id,
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": final_text, "annotations": []}],
                }
            ],
            "output_text": final_text,
        },
    }
    yield f"event: response.completed\ndata: {json.dumps(completed_evt, ensure_ascii=False)}\n\n"


def _stream_responses_tool_chunks(model: str, tool_calls: List[Dict[str, Any]]):
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    base_response = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
    }

    created_evt = {"type": "response.created", "response": {**base_response, "status": "in_progress"}}
    yield f"event: response.created\ndata: {json.dumps(created_evt, ensure_ascii=False)}\n\n"

    in_progress_evt = {"type": "response.in_progress", "response": {**base_response, "status": "in_progress"}}
    yield f"event: response.in_progress\ndata: {json.dumps(in_progress_evt, ensure_ascii=False)}\n\n"

    output_items = []
    for idx, call in enumerate(tool_calls):
        item = {
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "type": "function_call",
            "status": "completed",
            "call_id": f"call_{uuid.uuid4().hex[:24]}",
            "name": call["name"],
            "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False),
        }
        output_items.append(item)
        added_evt = {
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": idx,
            "item": item,
        }
        yield f"event: response.output_item.added\ndata: {json.dumps(added_evt, ensure_ascii=False)}\n\n"
        done_evt = {
            "type": "response.output_item.done",
            "response_id": response_id,
            "output_index": idx,
            "item": item,
        }
        yield f"event: response.output_item.done\ndata: {json.dumps(done_evt, ensure_ascii=False)}\n\n"

    completed_evt = {
        "type": "response.completed",
        "response": {**base_response, "status": "completed", "output": output_items, "output_text": ""},
    }
    yield f"event: response.completed\ndata: {json.dumps(completed_evt, ensure_ascii=False)}\n\n"


def create_app(llm: QwenService, served_model: str) -> FastAPI:
    app = FastAPI(title="AI Qwen OpenAI-Compatible API", version="0.1.0")

    def _try_zstd_decompress(data: bytes) -> bytes:
        if _zstd_builtin is not None:
            try:
                return _zstd_builtin.decompress(data)
            except Exception:
                pass
        if _zstd_ext is not None:
            try:
                return _zstd_ext.ZstdDecompressor().decompress(data)
            except Exception:
                pass
        return data

    def _decode_json_bytes(data: bytes) -> Dict[str, Any]:
        return json.loads(data.decode("utf-8"))

    async def _read_json_body(request: Request) -> Dict[str, Any]:
        raw = await request.body()
        if not raw:
            return {}

        # 1) 先按原始 JSON 解析
        try:
            return _decode_json_bytes(raw)
        except Exception:
            pass

        # 2) 按 content-encoding 明确解压
        encoding = (request.headers.get("content-encoding") or "").lower().strip()
        candidates: List[bytes] = []
        if encoding in {"gzip", "x-gzip"}:
            candidates.append(gzip.decompress(raw))
        elif encoding == "deflate":
            candidates.append(zlib.decompress(raw))
        elif encoding == "zstd":
            candidates.append(_try_zstd_decompress(raw))

        # 3) 魔数兜底识别（客户端可能没带 content-encoding）
        if raw.startswith(b"\x1f\x8b"):
            candidates.append(gzip.decompress(raw))
        if len(raw) >= 2 and raw[:1] == b"\x78":
            try:
                candidates.append(zlib.decompress(raw))
            except Exception:
                pass
        if raw.startswith(b"\x28\xb5\x2f\xfd"):
            candidates.append(_try_zstd_decompress(raw))

        for data in candidates:
            if not data:
                continue
            try:
                return _decode_json_bytes(data)
            except Exception:
                continue

        raise ValueError("Unsupported or invalid body encoding")

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

    @app.post("/v1/responses")
    async def responses(request: Request):
        try:
            payload = await _read_json_body(request)
            req = ResponsesRequest.model_validate(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid request body: {exc}")

        if req.model != served_model:
            raise HTTPException(
                status_code=400,
                detail=f"model '{req.model}' 不可用，可用模型: '{served_model}'",
            )

        messages = _responses_input_to_messages(req.input)
        prompt_tokens = llm.get_token_count(str(messages))

        if req.stream:
            content = llm.generate_response(
                messages=messages,
                tools=req.tools,
                max_new_tokens=req.max_output_tokens,
            )
            tool_calls = _extract_tool_calls(content)
            if tool_calls:
                return StreamingResponse(
                    _stream_responses_tool_chunks(served_model, tool_calls),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            stream_gen = (part for part in [content])
            return StreamingResponse(
                _stream_responses_chunks(served_model, stream_gen),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        content = llm.generate_response(
            messages=messages,
            tools=req.tools,
            max_new_tokens=req.max_output_tokens,
        )
        tool_calls = _extract_tool_calls(content)
        completion_tokens = llm.get_token_count(content)
        if tool_calls:
            return JSONResponse(
                _build_responses_tool_json(
                    model=served_model,
                    tool_calls=tool_calls,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        return JSONResponse(
            _build_responses_json(
                model=served_model,
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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
