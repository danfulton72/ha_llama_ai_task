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
    _preferred_model_id,
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


def test_preferred_model_requires_a_strong_signal() -> None:
    assert _preferred_model_id([{"id": "only"}]) == "only"
    assert (
        _preferred_model_id(
            [
                {"id": "first", "status": {"value": "unloaded"}},
                {"id": "loaded", "status": {"value": "loaded"}},
            ]
        )
        == "loaded"
    )
    assert (
        _preferred_model_id(
            [
                {"id": "first", "status": {"value": "unloaded"}},
                {"id": "second", "status": {"value": "unloaded"}},
            ]
        )
        is None
    )


@pytest.mark.asyncio
async def test_client_http_round_trip_and_router_discovery() -> None:
    state: dict[str, object] = {
        "router": False,
        "records": [{"id": "unsloth/Qwen3-8B-GGUF"}],
        "routed_props": [],
    }
    last_body: dict = {}

    async def props(request: web.Request) -> web.Response:
        model = request.query.get("model")
        if state["router"] and not model:
            return web.json_response(
                {"error": {"message": "model is required"}}, status=404
            )
        if model:
            cast_calls = state["routed_props"]
            assert isinstance(cast_calls, list)
            cast_calls.append((model, request.query.get("autoload")))
            records = state["records"]
            assert isinstance(records, list)
            record = next(
                (item for item in records if item.get("id") == model), None
            )
            if not record or record.get("status", {}).get("value") != "loaded":
                return web.json_response(
                    {"error": {"message": "model is not loaded"}}, status=404
                )
        return web.json_response(PROPS)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"data": state["records"]})

    async def chat(request: web.Request) -> web.Response:
        body = await request.json()
        last_body.clear()
        last_body.update(body)
        if body.get("model") == "missing":
            return web.json_response(
                {"error": {"message": "Model not found"}}, status=404
            )
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
            assert client.default_model == "unsloth/Qwen3-8B-GGUF"

            v1_client = LlamaCppClient(session, f"{base_url}/v1")
            assert v1_client.base_url == base_url
            assert (await v1_client.async_detect_server_info()).build_info == "b8681"

            # Generic router records deliberately do not contain llama-swap
            # markers. A loaded model is enough to route /props safely.
            state["router"] = True
            state["records"] = [
                {
                    "id": "qwen3.5-9b",
                    "owned_by": "llamacpp",
                    "status": {"value": "unloaded"},
                },
                {
                    "id": "qwen3.5-4b",
                    "owned_by": "llamacpp",
                    "status": {"value": "loaded"},
                },
                {
                    "id": "small",
                    "owned_by": "llamacpp",
                    "status": {"value": "unloaded"},
                },
            ]
            routed = LlamaCppClient(session, base_url)
            router_info = await routed.async_detect_server_info()
            assert router_info.n_ctx == 32768
            assert routed.default_model == "qwen3.5-4b"
            assert state["routed_props"] == [("qwen3.5-4b", "false")]

            payload = {"messages": [{"role": "user", "content": "hello"}]}
            result = await routed.async_chat_completion(payload)
            assert result["choices"][0]["message"]["content"] == "ok"
            assert last_body["model"] == "qwen3.5-4b"
            assert "model" not in payload

            # With several unloaded models, discovery accepts the routed model
            # catalogue but must not trigger a cold-start /props probe or choose
            # an arbitrary first model.
            state["records"] = [
                {"id": "one", "status": {"value": "unloaded"}},
                {"id": "two", "status": {"value": "unloaded"}},
            ]
            state["routed_props"] = []
            cold = LlamaCppClient(session, base_url)
            cold_info = await cold.async_detect_server_info()
            assert cold_info.props == {}
            assert cold.default_model is None
            assert state["routed_props"] == []
            assert await cold.async_list_models() == ["one", "two"]
            assert cold.default_model is None

            # A sole routed model is a safe request default but is still not
            # cold-started merely to inspect /props during setup.
            state["records"] = [
                {"id": "only", "status": {"value": "unloaded"}}
            ]
            state["routed_props"] = []
            sole = LlamaCppClient(session, base_url)
            await sole.async_detect_server_info()
            assert sole.default_model == "only"
            assert state["routed_props"] == []

            with pytest.raises(LlamaCppResponseError, match="Model not found"):
                await client.async_chat_completion(
                    {"model": "missing", "messages": []}
                )

            with pytest.raises(LlamaCppConnectionError, match="Timed out"):
                await client.async_chat_completion(
                    {"max_tokens": 1, "messages": []}, timeout=0.05
                )

            bad = LlamaCppClient(session, f"{base_url}/bad")
            with pytest.raises(LlamaCppResponseError, match="non-JSON"):
                await bad.async_get_server_info()
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_optional_api_key_is_sent_and_auth_errors_are_distinct() -> None:
    """An API key is optional, but when set it is sent as Bearer auth."""
    seen_headers: list[str | None] = []

    async def props(request: web.Request) -> web.Response:
        auth = request.headers.get("Authorization")
        seen_headers.append(auth)
        if auth != "Bearer secret":
            return web.json_response(
                {"error": {"message": "Unauthorized"}}, status=401
            )
        return web.json_response(PROPS)

    app = web.Application()
    app.router.add_get("/props", props)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    try:
        async with aiohttp.ClientSession() as session:
            without_key = LlamaCppClient(session, base_url)
            with pytest.raises(LlamaCppAuthError, match="Unauthorized"):
                await without_key.async_get_server_info()

            with_key = LlamaCppClient(session, base_url, "secret")
            assert (await with_key.async_get_server_info()).build_info == "b8681"

        assert seen_headers == [None, "Bearer secret"]
    finally:
        await runner.cleanup()
