"""Minimal async client for llama.cpp HTTP servers and llama-swap.

Only the endpoints needed by this integration are implemented:

* ``GET  /props``               - server introspection (model, build, modalities)
* ``GET  /v1/models``           - list servable models
* ``POST /v1/chat/completions`` - OpenAI-compatible chat completion

Plain ``llama-server`` exposes ``/props`` directly. ``llama-swap`` routes the
same endpoint only after a model is identified, so discovery falls back to the
loaded model reported by ``/v1/models`` and retries ``/props?model=<id>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from json import JSONDecodeError, loads as json_loads
from typing import Any

import aiohttp

from .const import DEFAULT_TIMEOUT, LOGGER, PROPS_TIMEOUT


class LlamaCppError(Exception):
    """Base error for the llama.cpp client."""


class LlamaCppConnectionError(LlamaCppError):
    """Raised when the server cannot be reached."""


class LlamaCppResponseError(LlamaCppError):
    """Raised when the server returns an error payload."""


@dataclass(slots=True)
class LlamaCppServerInfo:
    """Parsed view over the ``/props`` payload.

    Field names in ``/props`` have changed over llama.cpp releases, so every
    accessor degrades gracefully to ``None``/``False`` on older builds.
    """

    props: dict[str, Any] = field(default_factory=dict)

    @property
    def model_name(self) -> str | None:
        """Return a human readable name for the loaded model.

        Depending on how the server was started this is a Hugging Face repo id
        (``unsloth/Qwen3-8B-GGUF``) or a filesystem path. Paths are reduced to
        the file name; repo ids are left alone.
        """
        for key in ("model_alias", "model_name", "model_path", "model"):
            value = self.props.get(key)
            if not isinstance(value, str) or not value:
                continue
            if value.lower().endswith(".gguf"):
                return value.replace("\\", "/").rsplit("/", 1)[-1]
            return value
        return None

    @property
    def build_info(self) -> str | None:
        """Return the llama.cpp build tag, e.g. ``b8681-Debian``."""
        value = self.props.get("build_info")
        return value if isinstance(value, str) else None

    @property
    def n_ctx(self) -> int | None:
        """Return the context size of the default slot."""
        settings = self.props.get("default_generation_settings")
        if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
            return settings["n_ctx"]
        return None

    @property
    def modalities(self) -> dict[str, Any]:
        """Return the reported input modalities."""
        value = self.props.get("modalities")
        return value if isinstance(value, dict) else {}

    @property
    def supports_vision(self) -> bool:
        """Return True when the server was started with a vision projector."""
        return bool(self.modalities.get("vision"))

    @property
    def supports_audio(self) -> bool:
        """Return True when the server was started with an audio projector."""
        return bool(self.modalities.get("audio"))


def normalize_base_url(base_url: str) -> str:
    """Normalize a llama.cpp URL to the server root.

    Users commonly paste the OpenAI-compatible ``.../v1`` base URL. llama.cpp
    exposes introspection endpoints such as ``/props`` one level above it, so
    accept either spelling and store/use the root consistently.
    """
    normalized = base_url.rstrip("/")
    if normalized.lower().endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


class LlamaCppClient:
    """Thin wrapper around the llama.cpp server HTTP API."""

    def __init__(self, session: aiohttp.ClientSession, base_url: str) -> None:
        """Initialize the client."""
        self._session = session
        self._base_url = normalize_base_url(base_url)
        self._default_model: str | None = None

    @property
    def base_url(self) -> str:
        """Return the normalized base URL of the server."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """Return the model automatically selected for a routed server."""
        return self._default_model

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Any:
        """Perform a request and return the decoded JSON body."""
        url = f"{self._base_url}{path}"
        try:
            async with self._session.request(
                method,
                url,
                json=json,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                body = await response.text()
                if response.status >= 400:
                    raise LlamaCppResponseError(
                        _error_message(body, response.status, str(response.url))
                    )
                try:
                    return await response.json(content_type=None)
                except ValueError as err:
                    raise LlamaCppResponseError(
                        f"{response.url} returned a non-JSON response: {body[:200]}"
                    ) from err
        except TimeoutError as err:
            raise LlamaCppConnectionError(
                f"Timed out after {timeout:.0f}s waiting for {url}"
            ) from err
        except aiohttp.ClientError as err:
            raise LlamaCppConnectionError(f"Error talking to {url}: {err}") from err

    async def async_get_server_info(self, model: str | None = None) -> LlamaCppServerInfo:
        """Fetch ``/props`` and return the parsed server info."""
        params = {"model": model} if model else None
        props = await self._request(
            "GET", "/props", params=params, timeout=PROPS_TIMEOUT
        )
        if not isinstance(props, dict):
            raise LlamaCppResponseError("Unexpected /props payload")
        return LlamaCppServerInfo(props)

    async def _async_list_model_records(self) -> list[dict[str, Any]]:
        """Return raw model records from the OpenAI-compatible model list."""
        result = await self._request("GET", "/v1/models", timeout=PROPS_TIMEOUT)
        if not isinstance(result, dict):
            return []
        data = result.get("data", [])
        if not isinstance(data, list):
            return []
        return [model for model in data if isinstance(model, dict)]

    async def async_list_models(self) -> list[str]:
        """Return the model ids the server is willing to serve.

        Plain ``llama-server`` reports the single loaded model; router mode and
        proxies such as llama-swap report several. Failure is not fatal - the
        model field is optional - so an empty list is returned on error.
        """
        try:
            records = await self._async_list_model_records()
        except LlamaCppError as err:
            LOGGER.debug("Could not list models: %s", err)
            return []
        return [
            model["id"]
            for model in records
            if isinstance(model.get("id"), str) and model["id"]
        ]

    async def async_detect_server_info(self) -> LlamaCppServerInfo:
        """Discover a direct llama-server or a model-routed llama-swap server."""
        try:
            info = await self.async_get_server_info()
        except LlamaCppResponseError as direct_error:
            try:
                records = await self._async_list_model_records()
            except LlamaCppError:
                raise direct_error

            swap_records = [model for model in records if _is_llama_swap_model(model)]
            if not swap_records:
                raise direct_error

            # Prefer a model that llama-swap already has loaded so integration
            # setup does not unnecessarily cold-start another model. If none is
            # loaded, use the first configured model and let llama-swap load it.
            swap_records.sort(key=lambda model: not _model_is_loaded(model))
            for model in swap_records:
                model_id = model.get("id")
                if not isinstance(model_id, str) or not model_id:
                    continue
                try:
                    info = await self.async_get_server_info(model_id)
                except LlamaCppError as err:
                    LOGGER.debug(
                        "Could not inspect llama-swap model %s: %s", model_id, err
                    )
                    continue
                self._default_model = model_id
                LOGGER.debug("Detected llama-swap; routing through model %s", model_id)
                return info

            raise direct_error

        self._default_model = None
        return info

    async def async_chat_completion(
        self, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        """Call ``/v1/chat/completions`` and return the raw response."""
        request_payload = payload
        if self._default_model and not payload.get("model"):
            # Do not mutate the entity's payload; callers may log/reuse it.
            request_payload = {**payload, "model": self._default_model}

        result = await self._request(
            "POST", "/v1/chat/completions", json=request_payload, timeout=timeout
        )
        if not isinstance(result, dict):
            raise LlamaCppResponseError("Unexpected chat completion payload")
        if error := result.get("error"):
            raise LlamaCppResponseError(str(error))
        return result


def _is_llama_swap_model(model: dict[str, Any]) -> bool:
    """Return whether a model record identifies llama-swap."""
    if model.get("owned_by") == "llama-swap":
        return True
    meta = model.get("meta")
    return isinstance(meta, dict) and isinstance(meta.get("llamaswap"), dict)


def _model_is_loaded(model: dict[str, Any]) -> bool:
    """Return whether llama-swap reports a model as already loaded."""
    status = model.get("status")
    return isinstance(status, dict) and status.get("value") == "loaded"


def _error_message(body: str, status: int, url: str) -> str:
    """Extract a useful message from a llama.cpp error body."""
    try:
        parsed = json_loads(body)
        error = parsed.get("error", parsed)
        if isinstance(error, dict) and (message := error.get("message")):
            return f"{url} returned {status}: {message}"
    except (JSONDecodeError, AttributeError, TypeError):
        pass
    return f"{url} returned {status}: {body[:200]}"
