"""Minimal async client for the llama.cpp HTTP server (``llama-server``).

Only the endpoints needed by this integration are implemented:

* ``GET  /props``               - server introspection (model, build, modalities)
* ``GET  /v1/models``           - list servable models
* ``POST /v1/chat/completions`` - OpenAI-compatible chat completion
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


class LlamaCppAuthError(LlamaCppError):
    """Raised when the server rejects the API key."""


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


class LlamaCppClient:
    """Thin wrapper around the llama.cpp server HTTP API."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        """Return the base URL of the server."""
        return self._base_url

    def _headers(self) -> dict[str, str]:
        """Return the request headers."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Any:
        """Perform a request and return the decoded JSON body."""
        url = f"{self._base_url}{path}"
        try:
            async with self._session.request(
                method,
                url,
                headers=self._headers(),
                json=json,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                body = await response.text()
                if response.status in (401, 403):
                    raise LlamaCppAuthError(
                        "The llama.cpp server rejected the API key"
                    )
                if response.status >= 400:
                    raise LlamaCppResponseError(
                        _error_message(body, response.status, url)
                    )
                try:
                    return await response.json(content_type=None)
                except ValueError as err:
                    raise LlamaCppResponseError(
                        f"{url} returned a non-JSON response: {body[:200]}"
                    ) from err
        except TimeoutError as err:
            raise LlamaCppConnectionError(
                f"Timed out after {timeout:.0f}s waiting for {url}"
            ) from err
        except aiohttp.ClientError as err:
            raise LlamaCppConnectionError(f"Error talking to {url}: {err}") from err

    async def async_get_server_info(self) -> LlamaCppServerInfo:
        """Fetch ``/props`` and return the parsed server info."""
        props = await self._request("GET", "/props", timeout=PROPS_TIMEOUT)
        if not isinstance(props, dict):
            raise LlamaCppResponseError("Unexpected /props payload")
        return LlamaCppServerInfo(props)

    async def async_list_models(self) -> list[str]:
        """Return the model ids the server is willing to serve.

        Plain ``llama-server`` reports the single loaded model; router mode and
        proxies such as llama-swap report several. Failure is not fatal - the
        model field is optional - so an empty list is returned on error.
        """
        try:
            result = await self._request("GET", "/v1/models", timeout=PROPS_TIMEOUT)
        except LlamaCppError as err:
            LOGGER.debug("Could not list models: %s", err)
            return []
        if not isinstance(result, dict):
            return []
        return [
            model["id"]
            for model in result.get("data", [])
            if isinstance(model, dict) and isinstance(model.get("id"), str)
        ]

    async def async_chat_completion(
        self, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        """Call ``/v1/chat/completions`` and return the raw response."""
        result = await self._request(
            "POST", "/v1/chat/completions", json=payload, timeout=timeout
        )
        if not isinstance(result, dict):
            raise LlamaCppResponseError("Unexpected chat completion payload")
        if error := result.get("error"):
            raise LlamaCppResponseError(str(error))
        return result


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
