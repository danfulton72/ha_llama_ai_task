"""HTTP client tests using an in-process fake llama.cpp server."""

from __future__ import annotations

import asyncio

import aiohttp
import pytest
from aiohttp import web

from custom_components.llama_cpp_ai_task import client as client_module
from custom_components.llama_cpp_ai_task.client import (
    LlamaCppAuthError,
    LlamaCppClient,
    LlamaCppConnectionError,
    LlamaCppResponseError,
    LlamaCppServerInfo,
    _error_message,
    _is_routing_refusal,
    _preferred_model_id,
    _probe_candidates,
    normalize_base_url,
)

# What llama.cpp router mode actually answers to a bare /props. The model_alias
# and model_path values are placeholders so web UIs do not break; they name no
# model. See tools/server/server-models.cpp, get_router_props.
ROUTER_PROPS = {
    "role": "router",
    "model_alias": "llama-server",
    "model_path": "none",
    "build_info": "b-router",
    "max_instances": 2,
    "models_autoload": False,
    "default_generation_settings": {"params": {}, "n_ctx": 0},
}

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


def test_router_props_placeholders_are_not_a_model() -> None:
    """Router placeholders must never become an entry or AI Task name."""
    router = LlamaCppServerInfo(ROUTER_PROPS)
    assert router.is_router is True
    assert router.model_name is None
    assert router.n_ctx is None
    assert router.build_info == "b-router"
    assert router.supports_vision is False

    assert LlamaCppServerInfo(PROPS).is_router is False


def test_routing_refusal_is_matched_on_the_body_only() -> None:
    """A hostname must not be able to vouch for the software behind it."""
    assert _is_routing_refusal(None) is False
    assert (
        _is_routing_refusal(
            LlamaCppResponseError(
                "boom", body='{"error":{"message":"model is not loaded"}}'
            )
        )
        is True
    )
    assert (
        _is_routing_refusal(
            LlamaCppResponseError(
                "boom", body='{"error":{"message":"model name is missing from the request"}}'
            )
        )
        is True
    )
    # The formatted message carries the URL; only the body may be evidence.
    assert (
        _is_routing_refusal(
            LlamaCppResponseError(
                _error_message('{"detail":"Not Found"}', 404, "http://llama-swap.lan/props"),
                body='{"detail":"Not Found"}',
            )
        )
        is False
    )


def test_probe_candidates_never_walk_unloaded_models() -> None:
    """Only loaded models, or a sole model, may be inspected during setup."""
    assert _probe_candidates([]) == []
    assert _probe_candidates([{"id": "only", "status": {"value": "unloaded"}}]) == [
        "only"
    ]
    assert _probe_candidates(
        [
            {"id": "a", "status": {"value": "unloaded"}},
            {"id": "b", "status": {"value": "unloaded"}},
        ]
    ) == []
    assert _probe_candidates(
        [
            {"id": "a", "status": {"value": "unloaded"}},
            {"id": "b", "status": {"value": "loaded"}},
            {"id": "c", "status": {"value": "loaded"}},
            {"id": "d", "status": {"value": "loaded"}},
        ]
    ) == ["b", "c"]


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
        "native_router": False,
        "records": [{"id": "unsloth/Qwen3-8B-GGUF"}],
        "routed_props": [],
    }
    last_body: dict = {}

    async def props(request: web.Request) -> web.Response:
        model = request.query.get("model")
        if state["native_router"] and not model:
            return web.json_response(ROUTER_PROPS)
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

    async def generic_props(_request: web.Request) -> web.Response:
        return web.json_response({"detail": "Not Found"}, status=404)

    async def generic_models(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "generic-model"}]})

    app = web.Application()
    app.router.add_get("/props", props)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/bad/props", html)
    app.router.add_get("/generic/props", generic_props)
    app.router.add_get("/generic/v1/models", generic_models)
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

            # Listing is pure: it must not establish a request default implicitly.
            assert await client.async_list_models() == ["unsloth/Qwen3-8B-GGUF"]
            assert client.default_model is None
            assert await client.async_refresh_models() == ["unsloth/Qwen3-8B-GGUF"]
            assert client.default_model == "unsloth/Qwen3-8B-GGUF"

            v1_client = LlamaCppClient(session, f"{base_url}/v1")
            assert v1_client.base_url == base_url
            assert (await v1_client.async_detect_server_info()).build_info == "b8681"

            # Native llama.cpp router mode returns router-level /props successfully.
            # Discovery should then enrich it from an already-loaded model instead
            # of silently losing n_ctx/modalities at the router boundary.
            state["native_router"] = True
            state["records"] = [
                {
                    "id": "native-loaded",
                    "owned_by": "llamacpp",
                    "status": {"value": "loaded"},
                }
            ]
            state["routed_props"] = []
            native = LlamaCppClient(session, base_url)
            native_info = await native.async_detect_server_info()
            assert native_info.n_ctx == 32768
            assert native_info.supports_vision is True
            assert native.default_model == "native-loaded"
            assert state["routed_props"] == [("native-loaded", "false")]

            # Generic router records deliberately do not contain llama-swap
            # markers. A successful model-specific /props probe proves llama.cpp.
            state["native_router"] = False
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
            state["routed_props"] = []
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

            # Refreshing must not downgrade a probe-verified default to whichever
            # model the catalogue happens to list as loaded first.
            state["records"] = [
                {"id": "other", "status": {"value": "loaded"}},
                {"id": "qwen3.5-4b", "status": {"value": "unloaded"}},
            ]
            assert await routed.async_refresh_models() == ["other", "qwen3.5-4b"]
            assert routed.default_model == "qwen3.5-4b"

            # Once the verified model disappears from the catalogue the client
            # falls back to the ordinary loaded/sole-model signal.
            state["records"] = [{"id": "other", "status": {"value": "loaded"}}]
            assert await routed.async_refresh_models() == ["other"]
            assert routed.default_model == "other"

            # With several unloaded models, a recognizable router /props error is
            # enough to identify the router, but discovery must not cold-start any.
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

            # A sole model gets one capability probe with autoload=false. If the
            # router reports it is not loaded, the router signal is still enough
            # to accept it without causing an intentional cold start.
            state["records"] = [
                {"id": "only", "status": {"value": "unloaded"}}
            ]
            state["routed_props"] = []
            sole = LlamaCppClient(session, base_url)
            sole_info = await sole.async_detect_server_info()
            assert sole_info.props == {}
            assert sole.default_model == "only"
            assert state["routed_props"] == [("only", "false")]

            # A generic OpenAI-compatible model catalogue does not by itself prove
            # that the endpoint is llama.cpp. Both direct and routed /props fail
            # without a llama.cpp/router signal, so setup must reject the server.
            generic = LlamaCppClient(session, f"{base_url}/generic")
            with pytest.raises(LlamaCppResponseError, match="Not Found"):
                await generic.async_detect_server_info()

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


@pytest.mark.asyncio
async def test_routed_probe_timeout_is_a_connection_error(monkeypatch) -> None:
    """A slow cold start is a connection problem, not an invalid server."""

    async def props(request: web.Request) -> web.Response:
        if not request.query.get("model"):
            # llama-swap's wording when /props is called without a model.
            return web.json_response(
                {"src": "llama-swap", "error": {"message": "no model id could be identified"}},
                status=400,
            )
        # llama-swap ignores autoload=false, so the sole model cold-starts here.
        await asyncio.sleep(0.5)
        return web.json_response(PROPS)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "only"}]})

    app = web.Application()
    app.router.add_get("/props", props)
    app.router.add_get("/v1/models", models)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]

    monkeypatch.setattr(client_module, "ROUTED_PROPS_TIMEOUT", 0.05)
    try:
        async with aiohttp.ClientSession() as session:
            client = LlamaCppClient(session, f"http://127.0.0.1:{port}")
            with pytest.raises(LlamaCppConnectionError):
                await client.async_detect_server_info()
    finally:
        await runner.cleanup()
