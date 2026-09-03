"""Run LlamaCppClient against a fake llama-server."""

import asyncio
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "hastub"))
sys.path.insert(0, str(HERE.parent / "custom_components"))

import aiohttp
from aiohttp import web

from llama_cpp_ai_task import client as mod

PROPS = {
    "model_alias": "unsloth/Qwen3-8B-GGUF",
    "build_info": "b8681",
    "modalities": {"vision": True, "audio": False},
    "default_generation_settings": {"n_ctx": 32768},
}

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


REQUIRE_KEY = {"value": False}
last_body = {}


async def props(request):
    if REQUIRE_KEY["value"] and request.headers.get("Authorization") != "Bearer secret":
        return web.json_response(
            {"error": {"code": 401, "message": "Invalid API Key"}}, status=401
        )
    return web.json_response(PROPS)


async def models(request):
    return web.json_response(
        {"data": [{"id": "unsloth/Qwen3-8B-GGUF", "owned_by": "llamacpp"}]}
    )


async def chat(request):
    body = await request.json()
    last_body.clear()
    last_body.update(body)
    if body.get("model") == "missing":
        return web.json_response(
            {"error": {"code": 404, "message": "Model not found"}}, status=404
        )
    if body.get("max_tokens") == 1:
        await asyncio.sleep(2)  # trigger the client timeout
    return web.json_response(
        {
            "object": "chat.completion",
            "model": "unsloth/Qwen3-8B-GGUF",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": '{"ok": true}'},
                }
            ],
            "system_fingerprint": "b8681",
        }
    )


async def not_llama(request):
    return web.Response(text="<html>hello</html>", content_type="text/html")


async def main():
    app = web.Application()
    app.router.add_get("/props", props)
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_get("/nope/props", not_llama)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 8099)
    await site.start()

    async with aiohttp.ClientSession() as session:
        c = mod.LlamaCppClient(session, "http://127.0.0.1:8099/")
        check("trailing slash stripped", c.base_url == "http://127.0.0.1:8099")

        info = await c.async_get_server_info()
        check("props parsed", info.model_name == "unsloth/Qwen3-8B-GGUF")
        check("vision detected", info.supports_vision is True)
        check("n_ctx", info.n_ctx == 32768)

        check("models listed", await c.async_list_models() == ["unsloth/Qwen3-8B-GGUF"])

        result = await c.async_chat_completion(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": {"type": "json_schema", "schema": {"type": "object"}},
            }
        )
        check("chat ok", result["choices"][0]["message"]["content"] == '{"ok": true}')
        check("payload arrived", last_body["messages"][0]["content"] == "hi")

        # 404 with a llama.cpp error body
        try:
            await c.async_chat_completion({"model": "missing", "messages": []})
            check("404 raises", False)
        except mod.LlamaCppResponseError as err:
            check("404 raises with message", "Model not found" in str(err), str(err))

        # timeout
        try:
            await c.async_chat_completion({"max_tokens": 1, "messages": []}, timeout=0.5)
            check("timeout raises", False)
        except mod.LlamaCppConnectionError as err:
            check("timeout raises", "Timed out" in str(err), str(err))

        # non-JSON response
        bad = mod.LlamaCppClient(session, "http://127.0.0.1:8099/nope")
        try:
            await bad.async_get_server_info()
            check("non-JSON raises", False)
        except mod.LlamaCppResponseError as err:
            check("non-JSON raises", "non-JSON" in str(err), str(err))

        # unreachable server
        dead = mod.LlamaCppClient(session, "http://127.0.0.1:1")
        try:
            await dead.async_get_server_info()
            check("refused raises", False)
        except mod.LlamaCppConnectionError:
            check("refused connection raises", True)

        # auth
        REQUIRE_KEY["value"] = True
        try:
            await c.async_get_server_info()
            check("401 raises auth error", False)
        except mod.LlamaCppAuthError:
            check("401 raises auth error", True)

        authed = mod.LlamaCppClient(session, "http://127.0.0.1:8099", "secret")
        info = await authed.async_get_server_info()
        check("api key accepted", info.build_info == "b8681")
        check("list models tolerates failure", await mod.LlamaCppClient(
            session, "http://127.0.0.1:1"
        ).async_list_models() == [])

    await runner.cleanup()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


sys.exit(asyncio.run(main()))
