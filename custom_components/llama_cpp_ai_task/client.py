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
from typing import Any, Final

import aiohttp

from .const import (
    DEFAULT_TIMEOUT,
    LOGGER,
    MAX_ROUTED_PROPS_PROBES,
    PROPS_TIMEOUT,
    ROUTED_PROPS_TIMEOUT,
)


# Response-body substrings produced only by a llama.cpp-family model router
# asked to serve /props without a routable model. "model name is missing from the
# request" and "model is not loaded" are llama.cpp router mode's own wording; the
# rest cover llama-swap and older spellings.
ROUTING_REFUSAL_SIGNALS: Final = (
    "llama-swap",
    "no model id could be identified",
    "model name is missing from the request",
    "model is required",
    "model id is required",
    "model_id is required",
    "model is not loaded",
    "model not loaded",
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
    def is_router(self) -> bool:
        """Return whether this payload came from a model router itself.

        llama.cpp router mode answers a bare ``/props`` with ``role: router``.
        That payload also carries placeholder ``model_alias``/``model_path``
        values ("llama-server"/"none") which exist only so web UIs do not break;
        they describe no model and must never reach entity or task names.
        """
        return self.props.get("role") == "router"

    @property
    def model_name(self) -> str | None:
        """Return a human readable name for the loaded model.

        Depending on how the server was started this is a Hugging Face repo id
        (``unsloth/Qwen3-8B-GGUF``) or a filesystem path. Paths are reduced to
        the file name; repo ids are left alone.
        """
        if self.is_router:
            return None
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
        """Return the context size of the default slot.

        Router-level payloads report a placeholder ``n_ctx`` of 0, which is not a
        context size, so any non-positive value is reported as unknown.
        """
        settings = self.props.get("default_generation_settings")
        if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
            return settings["n_ctx"] or None
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
        # True only when a successful /props?model=... proved this exact model.
        self._default_model_verified = False

    @property
    def base_url(self) -> str:
        """Return the normalized base URL of the server."""
        return self._base_url

    @property
    def default_model(self) -> str | None:
        """Return a model that is safe to use as the automatic request default."""
        return self._default_model

    def _set_default_model(self, model_id: str | None, *, verified: bool) -> None:
        """Record the automatic request model and how well it is evidenced."""
        self._default_model = model_id
        self._default_model_verified = verified and model_id is not None

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
        """Refresh model IDs and the automatic request default explicitly.

        A default that discovery proved with a successful ``/props`` probe is kept
        for as long as the server still lists it. Only an unproven guess is
        replaced, so refreshing can never downgrade to a model whose own probe
        failed.
        """
        try:
            records = await self._async_list_model_records()
        except LlamaCppError as err:
            LOGGER.debug("Could not refresh models: %s", err)
            return []

        model_ids = _model_ids(records)
        if not (self._default_model_verified and self._default_model in model_ids):
            self._set_default_model(_preferred_model_id(records), verified=False)
        return model_ids

    async def _async_probe_model_props(
        self, model_id: str
    ) -> tuple[LlamaCppServerInfo | None, LlamaCppResponseError | None]:
        """Inspect one routed model without ever asking a router to load it.

        Returns ``(info, None)`` when the model could be inspected and
        ``(None, refusal)`` when the router answered but declined. A refusal is
        useful evidence, so it is returned rather than raised. Authentication and
        transport failures propagate: they say nothing about what kind of server
        this is, and swallowing a timeout here would report a slow cold start as
        "not a llama.cpp server" instead of as a connection problem.
        """
        try:
            info = await self.async_get_server_info(
                model_id, autoload=False, timeout=ROUTED_PROPS_TIMEOUT
            )
        except LlamaCppResponseError as refusal:
            return None, refusal
        return info, None

    async def _async_discover_routed(
        self,
        records: list[dict[str, Any]],
        *,
        router_info: LlamaCppServerInfo | None,
        direct_error: LlamaCppResponseError | None,
    ) -> LlamaCppServerInfo | None:
        """Identify a model-routed llama.cpp endpoint from its model catalogue.

        Returns the best server info available, or ``None`` when the endpoint
        could not be shown to be llama.cpp. ``router_info`` is a router-level
        ``/props`` payload that already proves it; ``direct_error`` is the bare
        ``/props`` failure whose body may prove it instead.
        """
        model_ids = _model_ids(records)
        if not model_ids:
            return router_info

        # An OpenAI-compatible /v1/models response is not evidence on its own:
        # Ollama, LM Studio, vLLM and generic proxies all serve one.
        proven = router_info is not None or _is_routing_refusal(direct_error)

        for model_id in _probe_candidates(records):
            info, refusal = await self._async_probe_model_props(model_id)
            if info is not None:
                self._set_default_model(model_id, verified=True)
                LOGGER.debug("Detected routed llama.cpp model %s", model_id)
                return info
            proven = proven or _is_routing_refusal(refusal)
            LOGGER.debug("Routed model %s could not be inspected: %s", model_id, refusal)

        if not proven:
            return None

        # A verified router with nothing inspectable. Capabilities stay unknown
        # rather than cold-starting models to discover them.
        self._set_default_model(_preferred_model_id(records), verified=False)
        LOGGER.info(
            "Connected to a llama.cpp model router with no inspectable model "
            "(models=%d, default=%s). Vision, audio and context size stay unknown "
            "until a model is loaded and the entry is reloaded",
            len(model_ids),
            self._default_model,
        )
        return router_info or LlamaCppServerInfo()

    async def async_detect_server_info(self) -> LlamaCppServerInfo:
        """Discover a direct llama-server or model-routed llama.cpp endpoint.

        A generic OpenAI-compatible ``/v1/models`` response is not sufficient to
        identify llama.cpp. Routed mode is accepted only when a model-specific
        ``/props`` probe succeeds or the ``/props`` error itself carries a known
        llama.cpp/router routing signal.

        Discovery never walks a list of unloaded models. ``autoload=false`` is
        included for routers that honour it; llama-swap ignores that parameter.
        A sole model is the one safe exception: it is probed once so capabilities
        can be discovered without risking a sequence of cold starts. If that probe
        cannot reach the server the connection error propagates, so a slow cold
        start is reported and retried as a connection problem rather than being
        misreported as an invalid server.
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

            routed_info = await self._async_discover_routed(
                records, router_info=None, direct_error=direct_error
            )
            if routed_info is None:
                raise direct_error from None
            return routed_info

        if info.is_router:
            # Router mode answers a bare /props with router-level metadata only;
            # per-model modalities and context size need /props?model=... . Enrich
            # it from a loaded or sole model without walking cold ones.
            try:
                records = await self._async_list_model_records()
            except LlamaCppAuthError:
                raise
            except LlamaCppError as err:
                LOGGER.debug("Could not list native router models: %s", err)
                return info
            routed_info = await self._async_discover_routed(
                records, router_info=info, direct_error=None
            )
            return routed_info or info

        self._set_default_model(None, verified=False)
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


def _probe_candidates(records: list[dict[str, Any]]) -> list[str]:
    """Return the routed models that are safe to inspect during setup.

    Models a router reports as loaded cost nothing to inspect. A sole model is
    the only other candidate: probing it risks at most one cold start on routers
    that ignore ``autoload=false``, and it is the only way to learn a
    single-model router's capabilities. Everything else is left alone.
    """
    loaded = [
        model_id
        for record in records
        if _model_is_loaded(record)
        and isinstance((model_id := record.get("id")), str)
        and model_id
    ]
    if loaded:
        return loaded[:MAX_ROUTED_PROPS_PROBES]

    model_ids = _model_ids(records)
    return model_ids if len(model_ids) == 1 else []


def _is_routing_refusal(error: LlamaCppResponseError | None) -> bool:
    """Return whether a /props failure is a llama.cpp router declining to route.

    Only the response body is examined. The formatted message embeds the request
    URL, so matching it would let a host named ``llama-swap.lan`` vouch for
    whatever software happens to answer on it.
    """
    if error is None:
        return False
    body = (error.body or "").lower()
    return any(signal in body for signal in ROUTING_REFUSAL_SIGNALS)


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
