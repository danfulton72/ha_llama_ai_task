"""HTTP client tests using an in-process fake llama.cpp server."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web

from custom_components.llama_cpp_ai_task.client import (
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppResponseError,
    LlamaCppServerInfo,
    _error_message,
    normalize_base_url,
)

PROPS = {
    "model_alias": "unsloth/Qwen3-8B-GGUF",
    "build_info": "b8681",
    "modalities": {"vision": True, "audio": False},
    "default_generation_settings": {"n_ctx": 32768},
}


def test_server_info() -> None:
    info = LlamaCppServerInfo(PROPS)
    assert info.model_name == "unsloth/Qwen3-8B-GGUF"
    assert info.build_info == "b8681"
    assert info.n_ctx == 32768
    assert info.supports_vision is True
    assert info.supports_audio is False

    path_info = LlamaCppServerInfo({"model_path": "/models/model-Q4.gguf"})
    assert path_info.model_name == "model-Q4.gguf"


def test_normalize_base_url() -> None:
    assert normalize_base_url("http://server:8080/") == "http://server:8080"
    assert normalize_base_url("http://server:8080/v1") == "http://server:8080"
    assert normalize_base_url("http://server:8080/v1/") == "http://server:8080"
    assert normalize_base_url("http://server:8080/proxy/v1") == "http://server:8080/proxy"


def test_error_message() -> None:
    assert "Model not found" in _error_message(
        '{"error":{"message":"Model not found"}}', 404, "http://server"
    )
    assert "<html>bad</html>" in _error_message(
        "<html>bad</html>", 502, "http://server"
    )


@pytest.mark.asyncio
async def test_client_http_round_trip() -> None:
    require_key = False
    state: dict[str, object] = {"swap": False, "last_props_model": None}
    last_body: dict = {}

    async def props(request: web.Request) -> web.Response:
        if require_key:
            return web.json_response({"error": {"message": "Unauthorized"}}, status=401)
        assert "Authorization" not in request.headers
        model = request.query.get("model")
        state["last_props_model"] = model
        if state["swap"] and not model:
            return web.json_response(
                {
                    "src": "llama-swap",
                    "error": {
                        "message": "no model id could be identified",
                        "type": "invalid_request_error",
                        "code": "not_found",
                    },
                },
                status=404,
            )
        if state["swap"] and model != "qwen3.5-4b":
            return web.json_response(
                {"error": {"message": "Model not found"}}, status=404
            )
        return web.json_response(PROPS)

    async def models(_request: web.Request) -> web.Response:
        if state["swap"]:
            return web.json_response(
                {
                    "data": [
                        {
                            "id": "qwen3.5-9b",
                            "owned_by": "llama-swap",
                            "meta": {"llamaswap": {"type": "model"}},
                            "status": {"value": "unloaded"},
                        },
                        {
                            "id": "qwen3.5-4b",
                            "owned_by": "llama-swap",
                            "meta": {"llamaswap": {"type": "model"}},
                            "status": {"value": "loaded"},
                        },
                        {
                            "id": "small",
                            "owned_by": "llama-swap",
                            "meta": {"llamaswap": {"type": "model"}},
                            "status": {"value": "unloaded"},
                        },
                    ]
                }
            )
        return web.json_response({"data": [{"id": "unsloth/Qwen3-8B-GGUF"}]})

    async def chat(request: web.Request) -> web.Response:
        assert "Authorization" not in request.headers
        body = await request.json()
        last_body.clear()
        last_body.update(body)
        if body.get("model") == "missing":
            return web.json_response({"error": {"message": "Model not found"}}, status=404)
        if body.get("max_tokens") == 1:
            await asyncio.sleep(0.2)
        return web.json_response(
            {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        )

    async def html(_request: web.Request) -> web.Response:
        return web.Response(text="<html>hello</html>", content_type="text/html")

    app = web.Application()
    app.router.add_get("/props", props)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/bad/props", html)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    try:
        async with aiohttp.ClientSession() as session:
            client = LlamaCppClient(session, f"{base_url}/")
            assert client.base_url == base_url
            assert (await client.async_detect_server_info()).n_ctx == 32768
            assert client.default_model is None
            assert await client.async_list_models() == ["unsloth/Qwen3-8B-GGUF"]

            # Pasting the OpenAI-compatible base URL must still target the same
            # llama.cpp server root rather than producing /v1/v1/... paths.
            v1_client = LlamaCppClient(session, f"{base_url}/v1")
            assert v1_client.base_url == base_url
            assert (await v1_client.async_detect_server_info()).build_info == "b8681"

            # Match llama-swap's real behaviour: bare /props cannot be routed,
            # /v1/models identifies llama-swap and reports a loaded model, and
            # /props?model=<id> reaches the underlying llama-server.
            state["swap"] = True
            swap_info = await client.async_detect_server_info()
            assert swap_info.n_ctx == 32768
            assert client.default_model == "qwen3.5-4b"
            assert state["last_props_model"] == "qwen3.5-4b"
            assert await client.async_list_models() == [
                "qwen3.5-9b",
                "qwen3.5-4b",
                "small",
            ]

            payload = {"messages": [{"role": "user", "content": "hello"}]}
            result = await client.async_chat_completion(payload)
            assert result["choices"][0]["message"]["content"] == "ok"
            assert last_body["messages"][0]["content"] == "hello"
            assert last_body["model"] == "qwen3.5-4b"
            assert "model" not in payload

            with pytest.raises(LlamaCppResponseError, match="Model not found"):
                await client.async_chat_completion({"model": "missing", "messages": []})

            with pytest.raises(LlamaCppConnectionError, match="Timed out"):
                await client.async_chat_completion(
                    {"max_tokens": 1, "messages": []}, timeout=0.05
                )

            bad = LlamaCppClient(session, f"{base_url}/bad")
            with pytest.raises(LlamaCppResponseError, match="non-JSON"):
                await bad.async_get_server_info()

            require_key = True
            with pytest.raises(LlamaCppResponseError, match="Unauthorized"):
                await client.async_get_server_info()
    finally:
        await runner.cleanup()
