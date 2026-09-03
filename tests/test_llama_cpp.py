"""Offline tests for the llama_cpp integration against Home Assistant stubs."""

import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "hastub"))
sys.path.insert(0, str(HERE.parent / "custom_components"))

import voluptuous as vol

from llama_cpp_ai_task import ai_task, client, config_flow, entity  # noqa: E402
from llama_cpp_ai_task.const import (  # noqa: E402
    CONF_ATTACHMENTS,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
)

from homeassistant.components import ai_task as ha_ai_task  # noqa: E402
from homeassistant.components import conversation  # noqa: E402
from homeassistant.config_entries import ConfigSubentry  # noqa: E402
from homeassistant.exceptions import HomeAssistantError  # noqa: E402

PASS = FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


PROPS = {
    "model_alias": "unsloth/Qwen3.6-35B-A3B-GGUF",
    "model_path": "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
    "total_slots": 4,
    "modalities": {"vision": True, "audio": False},
    "build_info": "b8681-Debian",
    "default_generation_settings": {"n_ctx": 65536},
}

print("\n== LlamaCppServerInfo ==")
info = client.LlamaCppServerInfo(PROPS)
check("model_name keeps repo id", info.model_name == "unsloth/Qwen3.6-35B-A3B-GGUF", info.model_name)
check("build_info", info.build_info == "b8681-Debian")
check("n_ctx", info.n_ctx == 65536)
check("vision", info.supports_vision is True)
check("audio", info.supports_audio is False)

empty = client.LlamaCppServerInfo()
check("empty degrades", empty.model_name is None and empty.supports_vision is False)

path_only = client.LlamaCppServerInfo({"model_path": "/models/gemma-3-4b-it-Q4.gguf"})
check("path basename", path_only.model_name == "gemma-3-4b-it-Q4.gguf", path_only.model_name)

print("\n== error message extraction ==")
msg = client._error_message(
    json.dumps({"error": {"code": 501, "message": "This server does not support embeddings"}}),
    501,
    "http://x/v1/chat/completions",
)
check("llama.cpp error body", "does not support embeddings" in msg, msg)
check("html body", "<html>" in client._error_message("<html>502</html>", 502, "u"))

print("\n== thinking strip ==")
check("closed block", entity.strip_thinking("<think>hmm</think>Answer").strip() == "Answer")
check(
    "multiline block",
    entity.strip_thinking("<think>\na\nb\n</think>\n{\"a\": 1}").strip() == '{"a": 1}',
)
check("forced open", entity.strip_thinking("reasoning here</think>Answer") == "Answer")
check("no block", entity.strip_thinking("Answer") == "Answer")
check("keeps html-ish text", entity.strip_thinking("<b>bold</b>") == "<b>bold</b>")

print("\n== response text extraction ==")
resp = {
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"is_full": true}',
                "reasoning_content": "Here's a thinking",
            },
        }
    ]
}
check("content", entity._extract_text(resp) == '{"is_full": true}')

try:
    entity._extract_text({"choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    check("token limit raises", False)
except HomeAssistantError as err:
    check("token limit raises", "token limit" in str(err), str(err))

try:
    entity._extract_text({"choices": []})
    check("no choices raises", False)
except HomeAssistantError:
    check("no choices raises", True)

typed = {"choices": [{"message": {"content": [{"type": "text", "text": "hi"}]}}]}
check("typed content parts", entity._extract_text(typed) == "hi")

print("\n== json isolation ==")
check("plain", ai_task._isolate_json('{"a": 1}') == '{"a": 1}')
check(
    "fenced",
    ai_task._isolate_json('```json\n{"a": 1}\n```') == '{"a": 1}',
    ai_task._isolate_json('```json\n{"a": 1}\n```'),
)
check(
    "prose wrapped",
    ai_task._isolate_json('Sure! {"a": 1} Hope that helps.') == '{"a": 1}',
)
check("array", ai_task._isolate_json('[{"a": 1}]') == '[{"a": 1}]')
check("no json passthrough", ai_task._isolate_json("nope") == "nope")

print("\n== structure -> json schema ==")
structure = vol.Schema(
    {
        vol.Required("summary"): str,
        vol.Optional("count"): int,
        vol.Optional("ok"): bool,
    }
)
schema = ai_task._to_json_schema(structure)
check("root type object", schema.get("type") == "object", schema)
check("has properties", set(schema.get("properties", {})) == {"summary", "count", "ok"}, schema)
check("required kept", schema.get("required") == ["summary"], schema)
check("json serialisable", json.dumps(schema) is not None)

dirty = {
    "type": "object",
    "properties": {
        "a": {"type": "string", "default": "x", "nullable": True},
        "b": {"type": "string", "enum": []},
        "c": {"type": "object"},
    },
}
cleaned = ai_task._clean_schema(dirty)
check("drops default", "default" not in cleaned["properties"]["a"])
check("drops nullable", "nullable" not in cleaned["properties"]["a"])
check("drops empty enum", "enum" not in cleaned["properties"]["b"])
check("drops propertyless object type", "type" not in cleaned["properties"]["c"], cleaned)

print("\n== payload + message building ==")


class FakeClient:
    def __init__(self):
        self.payload = None
        self.timeout = None
        self.base_url = "http://localhost:8080"

    async def async_chat_completion(self, payload, *, timeout):
        self.payload = payload
        self.timeout = timeout
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"summary": "ok"}'}}]}


class FakeRuntime:
    def __init__(self, fake):
        self.client = fake
        self.info = client.LlamaCppServerInfo(PROPS)


class FakeEntry:
    def __init__(self, fake):
        self.runtime_data = FakeRuntime(fake)
        self.entry_id = "e1"


fake = FakeClient()
subentry = ConfigSubentry(
    subentry_id="sub1",
    subentry_type="ai_task_data",
    title="Task",
    data={
        CONF_MAX_TOKENS: 512.0,
        CONF_TEMPERATURE: 0.2,
        CONF_PROMPT: "Always answer in metric units.",
        CONF_ATTACHMENTS: True,
    },
)
task_entity = ai_task.LlamaCppTaskEntity(FakeEntry(fake), subentry)

check(
    "features include attachments",
    task_entity.supported_features
    == ha_ai_task.AITaskEntityFeature.GENERATE_DATA
    | ha_ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS,
    task_entity.supported_features,
)

chat_log = conversation.ChatLog(
    [
        conversation.SystemContent("You are a Home Assistant expert."),
        conversation.UserContent("How many lights are on?"),
    ]
)
task = ha_ai_task.GenDataTask(name="t", instructions="How many lights are on?", structure=structure)

result = asyncio.run(task_entity._async_generate_data(task, chat_log))
payload = fake.payload

check("structured data parsed", result.data == {"summary": "ok"}, result.data)
check("conversation id", result.conversation_id == "conv1")
check("assistant appended to log", isinstance(chat_log.content[-1], conversation.AssistantContent))
check("max_tokens is int", payload["max_tokens"] == 512 and isinstance(payload["max_tokens"], int))
check("temperature passed", payload["temperature"] == 0.2)
check("stream off", payload["stream"] is False)
check("cache_prompt on", payload["cache_prompt"] is True)
check("thinking disabled", payload["chat_template_kwargs"] == {"enable_thinking": False})
check("no model key", "model" not in payload)
check("timeout default", fake.timeout == 120.0, fake.timeout)
check(
    "response_format has both shapes",
    payload["response_format"]["schema"]["type"] == "object"
    and payload["response_format"]["json_schema"]["schema"]["type"] == "object"
    and payload["response_format"]["type"] == "json_schema",
    payload["response_format"],
)

msgs = payload["messages"]
check("one system message", sum(m["role"] == "system" for m in msgs) == 1, msgs)
check("system merged with option prompt", "metric units" in msgs[0]["content"], msgs[0])
check("system contains ha prompt", "Home Assistant expert" in msgs[0]["content"])
check("user message follows", msgs[1] == {"role": "user", "content": "How many lights are on?"}, msgs)

print("\n== unstructured task returns text ==")
fake2 = FakeClient()


async def _text_response(payload, *, timeout):
    fake2.payload = payload
    return {"choices": [{"finish_reason": "stop", "message": {"content": "<think>x</think>Six lights."}}]}


fake2.async_chat_completion = _text_response
entity2 = ai_task.LlamaCppTaskEntity(FakeEntry(fake2), ConfigSubentry(data={}))
res2 = asyncio.run(
    entity2._async_generate_data(
        ha_ai_task.GenDataTask(structure=None), conversation.ChatLog([])
    )
)
check("plain text result", res2.data == "Six lights.", res2.data)
check("no response_format", "response_format" not in fake2.payload)
check(
    "features without vision option",
    entity2.supported_features
    == ha_ai_task.AITaskEntityFeature.GENERATE_DATA
    | ha_ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS,
    entity2.supported_features,
)

print("\n== attachments ==")
img = Path("/tmp/att.png")
img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
fake3 = FakeClient()
entity3 = ai_task.LlamaCppTaskEntity(FakeEntry(fake3), ConfigSubentry(data={}))
entity3.hass = type("H", (), {"async_add_executor_job": staticmethod(lambda f, *a: _run(f, *a))})()


async def _run(func, *args):
    return func(*args)


log3 = conversation.ChatLog(
    [
        conversation.UserContent(
            "What is in this image?",
            attachments=[
                conversation.Attachment(
                    media_content_id="media-source://x", mime_type="image/png", path=img
                )
            ],
        )
    ]
)
msgs3 = asyncio.run(entity3._async_build_messages(log3))
part = msgs3[0]["content"][1]
check("image part type", part["type"] == "image_url", part)
check("data url", part["image_url"]["url"].startswith("data:image/png;base64,"), part)
check("text part first", msgs3[0]["content"][0]["type"] == "text")

bad = conversation.Attachment(media_content_id="m", mime_type="application/pdf", path=img)
try:
    asyncio.run(entity3._async_attachment_part(bad))
    check("pdf rejected", False)
except HomeAssistantError as err:
    check("pdf rejected", "Unsupported attachment" in str(err), str(err))

wav = conversation.Attachment(media_content_id="m", mime_type="audio/wav", path=img)
audio_part = asyncio.run(entity3._async_attachment_part(wav))
check("wav format", audio_part["input_audio"]["format"] == "wav", audio_part)

print("\n== bad structured output ==")
fake4 = FakeClient()


async def _bad(payload, *, timeout):
    return {"choices": [{"finish_reason": "stop", "message": {"content": "I cannot do that."}}]}


fake4.async_chat_completion = _bad
entity4 = ai_task.LlamaCppTaskEntity(FakeEntry(fake4), ConfigSubentry(data={}))
try:
    asyncio.run(
        entity4._async_generate_data(
            ha_ai_task.GenDataTask(structure=structure), conversation.ChatLog([])
        )
    )
    check("invalid json raises", False)
except HomeAssistantError as err:
    check("invalid json raises", "requested structure" in str(err), str(err))

print("\n== config flow schema ==")
schema_new = config_flow._options_schema(["model-a", "model-b"], include_name=True)
keys = {str(k) for k in schema_new.schema}
check("name in new flow", "name" in keys, keys)
check("model offered", "model" in keys)
schema_re = config_flow._options_schema([], include_name=False)
check("no name on reconfigure", "name" not in {str(k) for k in schema_re.schema})
validated = schema_re({"max_tokens": 1024, "temperature": 0.4, "top_p": 0.95, "top_k": 40,
                       "repeat_penalty": 1.1, "timeout": 120, "thinking": False,
                       "attachments": True})
check("schema validates defaults", validated["max_tokens"] == 1024, validated)

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
