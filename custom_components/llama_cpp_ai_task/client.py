"""Minimal async client for llama.cpp HTTP servers and model routers.

Only the endpoints needed by this integration are implemented:

* ``GET  /props``               - server introspection (model, build, modalities)
* ``GET  /v1/models``           - list servable models
* ``POST /v1/chat/completions`` - OpenAI-compatible chat completion

Plain ``llama-server`` exposes ``/props`` directly. Routers such as llama-swap
and llama.cpp router mode may require a model query parameter for model-specific
introspection, so discovery can fall back to a model already reported as loaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from json import JSONDecodeError, loads as json_loads
from typing import Any

import aiohttp

from .const import (
    DEFAULT_TIMEOUT,
    LOGGER,
    MAX_ROUTED_PROPS_PROBES,
    PROPS_TIMEOUT,
    ROUTED_PROPS_TIMEOUT,
)


class LlamaCppError(Exception):
    """Base error for the llama.cpp client."""


class LlamaCppConnectionError(LlamaCppError):
    """Raised when the server cannot be reached."""


class LlamaCppAuthError(LlamaCppError):
    """Raised when the server rejects authentication."""


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
        """Return a human readable name for the loaded model."""
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
    """Normalize a llama.cpp URL to the server root."""
    normalized = base_url.rstrip("/")
    if normalized.lower().endswith("/v1"):
        normalized = normalized[:-3].rstrip("/")
    return normalized


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
        self._base_url = normalize_base_url(base_url)
        self._api_key = api_key or None
        self._default_model: str | None = None

    @property
    def base_url(self) -> str:
        """Return the normalized base URL of the server."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """Return a model that is safe to use as the automatic request default."""
        return self._default_model

    def _headers(self) -> dict[str, str] | None:
        """Return optional authentication headers."""
        if not self._api_key:
            return None
        return {"Authorization": f"Bearer {self._api_key}"}

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
                headers=self._headers(),
                json=json,
                params=params,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                body = await response.text()
                if response.status in (401, 403):
                    raise LlamaCppAuthError(
                        _error_message(body, response.status, str(response.url))
                    )
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

    async def async_get_server_info(
        self,
        model: str | None = None,
        *,
        autoload: bool | None = None,
        timeout: float = PROPS_TIMEOUT,
    ) -> LlamaCppServerInfo:
        """Fetch ``/props`` and return the parsed server info."""
        params: dict[str, str] = {}
        if model:
            params["model"] = model
        if autoload is not None:
            params["autoload"] = "true" if autoload else "false"
        props = await self._request(
            "GET", "/props", params=params or None, timeout=timeout
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
        """Return model IDs the endpoint reports as servable.

        A default is chosen only when the server gives us a meaningful signal:
        an already-loaded model or exactly one available model. Multi-model lists
        with no loaded/default signal are deliberately left unselected.
        """
        try:
            records = await self._async_list_model_records()
        except LlamaCppError as err:
            LOGGER.debug("Could not list models: %s", err)
            return []

        model_ids = _model_ids(records)
        if self._default_model is None:
            self._default_model = _preferred_model_id(records)
        return model_ids

    async def async_detect_server_info(self) -> LlamaCppServerInfo:
        """Discover a direct llama-server or model-routed endpoint.

        Routed discovery never intentionally probes an unloaded model. This avoids
        repeated cold starts during Home Assistant setup. ``autoload=false`` is
        included for routers that honour it; llama-swap ignores that parameter.
        """
        try:
            info = await self.async_get_server_info()
        except LlamaCppAuthError:
            raise
        except LlamaCppResponseError as direct_error:
            try:
                records = await self._async_list_model_records()
            except LlamaCppError:
                raise direct_error from None

            model_ids = _model_ids(records)
            if not model_ids:
                raise direct_error from None

            loaded_records = [record for record in records if _model_is_loaded(record)]
            for record in loaded_records[:MAX_ROUTED_PROPS_PROBES]:
                model_id = record.get("id")
                if not isinstance(model_id, str) or not model_id:
                    continue
                try:
                    info = await self.async_get_server_info(
                        model_id,
                        autoload=False,
                        timeout=ROUTED_PROPS_TIMEOUT,
                    )
                except LlamaCppAuthError:
                    raise
                except LlamaCppError as err:
                    LOGGER.debug(
                        "Could not inspect loaded routed model %s: %s", model_id, err
                    )
                    continue
                self._default_model = model_id
                LOGGER.debug("Detected routed llama.cpp model %s", model_id)
                return info

            # A valid model catalogue is enough to identify a routed endpoint even
            # when none of its models are currently loaded. Do not cold-start one
            # merely for capability discovery. A sole model is safe as the request
            # default; a multi-model endpoint remains unselected until the user or
            # server provides a loaded/default signal.
            self._default_model = _preferred_model_id(records)
            LOGGER.debug(
                "Detected model-routed endpoint without probing unloaded models "
                "(models=%d, default=%s)",
                len(model_ids),
                self._default_model,
            )
            return LlamaCppServerInfo()

        self._default_model = None
        return info

    async def async_chat_completion(
        self, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        """Call ``/v1/chat/completions`` and return the raw response."""
        request_payload = payload
        if self._default_model and not payload.get("model"):
            request_payload = {**payload, "model": self._default_model}

        result = await self._request(
            "POST", "/v1/chat/completions", json=request_payload, timeout=timeout
        )
        if not isinstance(result, dict):
            raise LlamaCppResponseError("Unexpected chat completion payload")
        if error := result.get("error"):
            raise LlamaCppResponseError(str(error))
        return result


def _model_ids(records: list[dict[str, Any]]) -> list[str]:
    """Return non-empty model IDs in server order."""
    return [
        model_id
        for record in records
        if isinstance((model_id := record.get("id")), str) and model_id
    ]


def _preferred_model_id(records: list[dict[str, Any]]) -> str | None:
    """Choose an automatic model only from a strong server-side signal."""
    for record in records:
        if _model_is_loaded(record):
            model_id = record.get("id")
            if isinstance(model_id, str) and model_id:
                return model_id

    model_ids = _model_ids(records)
    return model_ids[0] if len(model_ids) == 1 else None


def _model_is_loaded(model: dict[str, Any]) -> bool:
    """Return whether a router reports a model as already loaded."""
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
