"""HTTP client tests using an in-process fake llama.cpp server."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web

from custom_components.llama_cpp_ai_task.client import (
    LlamaCppAuthError,
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppResponseError,
    LlamaCppServerInfo,
    _error_message,
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
    last_body: dict = {}

    async def props(request: web.Request) -> web.Response:
        if require_key and request.headers.get("Authorization") != "Bearer secret":
            return web.json_response({"error": {"message": "Invalid API Key"}}, status=401)
        return web.json_response(PROPS)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "unsloth/Qwen3-8B-GGUF"}]})

    async def chat(request: web.Request) -> web.Response:
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
            assert (await client.async_get_server_info()).n_ctx == 32768
            assert await client.async_list_models() == ["unsloth/Qwen3-8B-GGUF"]
            result = await client.async_chat_completion(
                {"messages": [{"role": "user", "content": "hello"}]}
            )
            assert result["choices"][0]["message"]["content"] == "ok"
            assert last_body["messages"][0]["content"] == "hello"

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
            with pytest.raises(LlamaCppAuthError):
                await client.async_get_server_info()
            authed = LlamaCppClient(session, base_url, "secret")
            assert (await authed.async_get_server_info()).build_info == "b8681"
    finally:
        await runner.cleanup()
