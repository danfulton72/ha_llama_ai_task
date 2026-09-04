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

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str | None = None,
        url: str | None = None,
    ) -> None:
        """Initialize an HTTP response error with diagnostic response details."""
        super().__init__(message)
        self.status = status
        self.body = body
        self.url = url


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
                        _error_message(body, response.status, str(response.url)),
                        status=response.status,
                        body=body,
                        url=str(response.url),
                    )
                try:
                    return await response.json(content_type=None)
                except ValueError as err:
                    raise LlamaCppResponseError(
                        f"{response.url} returned a non-JSON response: {body[:200]}",
                        status=response.status,
                        body=body,
                        url=str(response.url),
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
        """Return model IDs the endpoint reports as servable without mutating state.

        Failure is not fatal because the model field can be entered manually. Use
        ``async_refresh_models`` when the caller also wants the client to refresh
        its automatic request default from a loaded/sole-model signal.
        """
        try:
            records = await self._async_list_model_records()
        except LlamaCppError as err:
            LOGGER.debug("Could not list models: %s", err)
            return []
        return _model_ids(records)

    async def async_refresh_models(self) -> list[str]:
        """Refresh model IDs and the automatic request default explicitly."""
        try:
            records = await self._async_list_model_records()
        except LlamaCppError as err:
            LOGGER.debug("Could not refresh models: %s", err)
            return []
        self._default_model = _preferred_model_id(records)
        return _model_ids(records)

    async def _async_probe_routed_models(
        self,
        records: list[dict[str, Any]],
        *,
        evidence_error: LlamaCppResponseError | None,
        fallback_info: LlamaCppServerInfo | None,
    ) -> LlamaCppServerInfo | None:
        """Safely inspect loaded/sole routed models and return verified info.

        ``fallback_info`` means the direct `/props` response already proved native
        llama.cpp router mode (currently reported as ``role: router``). Otherwise
        a successful model-specific `/props` response or a known routing error is
        required before a generic OpenAI-compatible model catalogue is accepted.
        """
        model_ids = _model_ids(records)
        if not model_ids:
            return fallback_info

        probed_ids: set[str] = set()
        loaded_records = [record for record in records if _model_is_loaded(record)]
        for record in loaded_records[:MAX_ROUTED_PROPS_PROBES]:
            model_id = record.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            probed_ids.add(model_id)
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

        if len(model_ids) == 1 and model_ids[0] not in probed_ids:
            model_id = model_ids[0]
            try:
                info = await self.async_get_server_info(
                    model_id,
                    autoload=False,
                    timeout=ROUTED_PROPS_TIMEOUT,
                )
            except LlamaCppAuthError:
                raise
            except LlamaCppResponseError as routed_error:
                verified = (
                    fallback_info is not None
                    or (
                        evidence_error is not None
                        and _looks_like_routed_props_error(evidence_error)
                    )
                    or _looks_like_routed_props_error(routed_error)
                )
                if not verified:
                    return None
                LOGGER.debug(
                    "Recognized routed llama.cpp endpoint with sole model %s; "
                    "model capabilities are not currently available",
                    model_id,
                )
            except LlamaCppError as err:
                if fallback_info is None:
                    LOGGER.debug("Could not verify sole routed model %s: %s", model_id, err)
                    return None
                LOGGER.debug("Could not inspect sole routed model %s: %s", model_id, err)
            else:
                self._default_model = model_id
                LOGGER.debug("Detected routed llama.cpp model %s", model_id)
                return info

            self._default_model = model_id
            return fallback_info or LlamaCppServerInfo()

        verified = fallback_info is not None or (
            evidence_error is not None and _looks_like_routed_props_error(evidence_error)
        )
        if not verified:
            return None

        # The router is verified but no loaded model could be inspected safely.
        # Keep capabilities unknown rather than cold-starting multiple models.
        self._default_model = _preferred_model_id(records)
        LOGGER.debug(
            "Detected llama.cpp model router without probing unloaded models "
            "(models=%d, default=%s)",
            len(model_ids),
            self._default_model,
        )
        return fallback_info or LlamaCppServerInfo()

    async def async_detect_server_info(self) -> LlamaCppServerInfo:
        """Discover a direct llama-server or model-routed llama.cpp endpoint.

        A generic OpenAI-compatible ``/v1/models`` response is not sufficient to
        identify llama.cpp. Routed mode is accepted only when a model-specific
        ``/props`` probe succeeds or the ``/props`` error itself carries a known
        llama.cpp/router routing signal.

        Discovery never walks a list of unloaded models. ``autoload=false`` is
        included for routers that honour it; llama-swap ignores that parameter.
        A sole model is the one safe exception: it is probed once so capabilities
        can be discovered without risking a sequence of cold starts.
        """
        try:
            info = await self.async_get_server_info()
        except LlamaCppAuthError:
            raise
        except LlamaCppResponseError as direct_error:
            try:
                records = await self._async_list_model_records()
            except LlamaCppAuthError:
                raise
            except LlamaCppError:
                raise direct_error from None

            routed_info = await self._async_probe_routed_models(
                records,
                evidence_error=direct_error,
                fallback_info=None,
            )
            if routed_info is None:
                raise direct_error from None
            return routed_info

        if info.props.get("role") == "router":
            # Native llama.cpp router mode exposes a successful router-level
            # /props payload containing build/router metadata, but per-model
            # modalities/context require /props?model=... . Safely enrich it when
            # a loaded or sole model can be inspected without walking cold models.
            try:
                records = await self._async_list_model_records()
            except LlamaCppAuthError:
                raise
            except LlamaCppError as err:
                LOGGER.debug("Could not list native router models: %s", err)
                return info
            routed_info = await self._async_probe_routed_models(
                records,
                evidence_error=None,
                fallback_info=info,
            )
            return routed_info or info

        self._default_model = None
        return info

    async def async_chat_completion(
        self, payload: dict[str, Any], *, timeout: float = DEFAULT_TIMEOUT
    ) -> dict[str, Any]:
        """Call ``/v1/chat/completions`` and return the raw response."""
        # Do not mutate the entity's payload when supplying an automatic router
        # model; callers may reuse/debug that payload after this method returns.
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
    """Return whether a known router schema reports a model as already loaded."""
    status = model.get("status")
    return isinstance(status, dict) and status.get("value") == "loaded"


def _looks_like_routed_props_error(error: LlamaCppResponseError) -> bool:
    """Return whether a /props failure carries a model-router routing signal."""
    text = f"{error.body or ''}\n{error}".lower()
    return (
        "llama-swap" in text
        or "no model id could be identified" in text
        or "model is required" in text
        or "model id is required" in text
        or "model_id is required" in text
        or "model is not loaded" in text
        or "model not loaded" in text
    )


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
