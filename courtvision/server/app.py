"""FastAPI app: REST + WebSocket API and the static operator/coach dashboard.

``create_app(hub, controller=None, auth_token=None)`` builds the app described in
ARCHITECTURE "server/". Reads come from the :class:`~courtvision.server.hub.Hub`; writes go
to the pipeline through the :class:`Controller` protocol (503 when none is attached).

Security model (local, in-gym server):
* With ``auth_token`` set, every ``/api/*`` request needs ``Authorization: Bearer <token>``
  (POSTs always; GETs too unless ``ServerConfig.protect_reads=False``, because responses carry
  rosters of minors) and ``/ws`` needs ``?token=<token>``. Compared in constant time.
* Browser requests with a foreign ``Origin`` are refused on unsafe methods and on ``/ws``
  (cross-site WebSocket hijacking / CSRF against a localhost server).
* Requests whose ``Host`` is a public DNS name are refused unless listed in
  ``ServerConfig.allowed_hosts`` (DNS rebinding would otherwise pass the same-origin check).
* Manual events are checked against the shared event semantics: per-kind ``value`` ranges,
  ``team`` = the actor's team (filled in from the roster or id prefix when omitted), and the
  teammate/opponent relation of ``other_player_id``.
* Static files are served by Starlette ``StaticFiles`` (rejects ``..`` traversal and symlinks
  out of the directory); responses carry a strict same-origin CSP.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import ipaddress
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from courtvision.server.hub import Hub, HubClient, dumps, to_jsonable
from courtvision.types import EventKind, EventStatus, Team

log = logging.getLogger(__name__)

__all__ = ["Controller", "ServerConfig", "create_app"]

STATIC_DIR = Path(__file__).resolve().parent / "static"


# --------------------------------------------------------------------------- config / protocol


@dataclass
class ServerConfig:
    """Server options (engineering defaults; status: tuned)."""

    static_dir: str = ""  # "" -> the package's server/static directory
    protect_reads: bool = True  # with a token, GET /api/* needs it too (rosters of minors, D16 spirit)
    allowed_origins: list[str] = field(default_factory=list)  # extra browser origins; same-origin always ok
    # Extra Host names accepted (exact or "*.suffix"; "*" disables the check). IP literals,
    # single-label names (localhost, a Mac's bare hostname) and private-use suffixes (.local,
    # .localhost, .lan, .home.arpa, .internal) are always accepted. Anything else is refused so
    # a DNS-rebinding page cannot pass the same-origin check against this local server.
    allowed_hosts: list[str] = field(default_factory=list)
    max_body_bytes: int = 65_536  # largest accepted request body (calibration/ROI payloads are < 8 KB)
    events_limit_default: int = 1_000  # GET /api/events default page size
    events_limit_max: int = 20_000  # matches HubConfig.event_log_max
    ws_receive_max_bytes: int = 4_096  # client->server WS messages (pings) larger than this are ignored
    max_speed: float = 16.0  # largest playback speed accepted by POST /api/control
    openapi: bool = True  # serve /openapi.json (the CDN-based /docs UI is never served: offline)
    telemetry: bool = False  # FastAPI native OpenTelemetry (off: footage-adjacent data stays local)


@runtime_checkable
class Controller(Protocol):
    """Pipeline-side handler for operator writes. Methods run on the server's event loop and may
    return awaitables. ``ValueError``/``LookupError`` map to HTTP 400."""

    def add_manual(self, event: dict[str, Any]) -> Any: ...

    def set_scoreboard(self, reading: dict[str, Any]) -> Any: ...

    def calibrate(self, request: dict[str, Any]) -> dict[str, Any]: ...

    def control(self, command: dict[str, Any]) -> Any: ...

    def latest_frame_jpeg(self) -> bytes | None: ...

    def set_lineup(self, entry: dict[str, Any]) -> Any: ...

    def set_scoreboard_roi(self, roi: dict[str, Any]) -> Any: ...


class HubJSONResponse(JSONResponse):
    """JSONResponse that tolerates numpy, enums, dataclasses and NaN (-> null)."""

    def render(self, content: Any) -> bytes:
        return dumps(content).encode("utf-8")


# --------------------------------------------------------------------------- request models

_ID = r"^[A-Za-z0-9_.:\-]{1,64}$"
_CELL = r"^[A-Za-z0-9_.\-]{1,32}$"
_LANDMARK = r"^[A-Za-z0-9_.:\- ]{1,64}$"
Point = Annotated[list[float], Field(min_length=2, max_length=2)]
Box = Annotated[list[float], Field(min_length=4, max_length=4)]
ImageSize = Annotated[list[Annotated[int, Field(ge=1, le=16_384)]], Field(min_length=2, max_length=2)]
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)

# Event.value meaning per kind (types.py; stats/boxscore.py): shots carry 1-3 points, a
# SCORE_CORRECTION carries a signed point delta (stats adds it to the team), PERIOD_START /
# PERIOD_END may carry the period number (0 = unspecified). Everything else uses 0-3.
_SHOT_VALUE_KINDS = frozenset({EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE})
_PERIOD_KINDS = frozenset({EventKind.PERIOD_START, EventKind.PERIOD_END})
_FG_SHOT_TYPES = frozenset({"jumper", "layup", "dunk", "hook", "tip"})
# Kinds whose actor is player_id (types.py EventKind comments; ARCHITECTURE "Shared event semantics").
_NEEDS_ACTOR = frozenset({EventKind.PASS, EventKind.ASSIST, EventKind.STEAL, EventKind.BLOCK,
                          EventKind.SUBSTITUTION, EventKind.SHOT_ATTEMPT, EventKind.SHOT_MADE,
                          EventKind.SHOT_MISSED, EventKind.FOUL})
# Kinds that may have no team at all (an official / media timeout has none).
_TEAMLESS_OK = frozenset({EventKind.PERIOD_START, EventKind.PERIOD_END, EventKind.TIMEOUT})
# other_player_id relation to player_id: assister/scorer and sub in/out are teammates; stealer/
# victim, blocker/shooter and fouler/fouled are opponents. PASS is left open: an intercepted pass
# names the intended receiver or the interceptor (Shared event semantics).
_SAME_TEAM_PAIR = frozenset({EventKind.ASSIST, EventKind.SUBSTITUTION})
_OPPONENT_PAIR = frozenset({EventKind.STEAL, EventKind.BLOCK, EventKind.FOUL})


class ManualEventIn(BaseModel):
    """Operator insert (no ``event_id``) or correction (``event_id`` of the event to revise)."""

    model_config = _STRICT

    event_id: str | None = Field(None, pattern=_ID)
    kind: EventKind
    t: float = Field(ge=0)
    frame_idx: int = Field(0, ge=0)
    team: Team | None = None
    player_id: str | None = Field(None, pattern=_ID)
    other_player_id: str | None = Field(None, pattern=_ID)
    x: float | None = Field(None, ge=-20, le=200)  # court ft, generous margin past any baseline
    y: float | None = Field(None, ge=-20, le=120)
    value: int = Field(0, ge=-999, le=999)  # range per kind is checked in _semantics
    shot_id: str | None = Field(None, pattern=_ID)
    confidence: float = Field(1.0, ge=0, le=1)
    status: EventStatus = EventStatus.CONFIRMED
    revision: int = Field(0, ge=0)
    attrs: dict[str, Any] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def _semantics(self) -> ManualEventIn:
        if self.status is EventStatus.PROVISIONAL:
            raise ValueError("operator events are confirmed or retracted, never provisional")
        if len(dumps(self.attrs)) > 4096:
            raise ValueError("attrs too large")
        if self.status is EventStatus.RETRACTED:
            if not self.event_id:
                raise ValueError("a retraction needs the event_id it retracts")
            return self
        self._check_value()
        if self.kind in _NEEDS_ACTOR and not self.player_id:
            raise ValueError(f"{self.kind.value} needs player_id")
        if self.player_id and self.player_id == self.other_player_id:
            raise ValueError("player_id and other_player_id must differ")
        return self

    def _check_value(self) -> None:
        kind, value = self.kind, self.value
        if kind in _SHOT_VALUE_KINDS:
            if value not in (1, 2, 3):
                raise ValueError(f"{kind.value} needs value 1, 2 or 3")
            shot_type = self.attrs.get("shot_type")
            if shot_type == "free_throw" and value != 1:
                raise ValueError("a free throw is worth 1 point")
            if shot_type in _FG_SHOT_TYPES and value == 1:
                raise ValueError(f"a {shot_type} field goal is worth 2 or 3 points")
        elif kind is EventKind.SCORE_CORRECTION:
            if value == 0:
                raise ValueError("score_correction needs a non-zero signed point delta in value")
        elif kind in _PERIOD_KINDS:
            if not 0 <= value <= 20:
                raise ValueError(f"{kind.value} value is the period number (0-20)")
        elif not 0 <= value <= 3:
            raise ValueError(f"{kind.value} needs value 0-3")

    def to_controller(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["source"] = "manual"
        return d


class ScoreboardIn(BaseModel):
    """Operator score/clock override (source ``manual``)."""

    model_config = _STRICT

    t: float | None = Field(None, ge=0)
    home_score: int | None = Field(None, ge=0, le=999)
    away_score: int | None = Field(None, ge=0, le=999)
    period: int | None = Field(None, ge=1, le=20)
    game_clock: float | None = Field(None, ge=0, le=3600)
    shot_clock: float | None = Field(None, ge=0, le=60)

    @model_validator(mode="after")
    def _non_empty(self) -> ScoreboardIn:
        fields = (self.home_score, self.away_score, self.period, self.game_clock, self.shot_clock)
        if all(v is None for v in fields):
            raise ValueError("set at least one of home_score, away_score, period, game_clock, shot_clock")
        return self

    def to_controller(self) -> dict[str, Any]:
        d = self.model_dump(mode="json", exclude_none=True)
        d["source"] = "manual"
        return d


class LineupIn(BaseModel):
    """Tap-to-sub ``{"team","out","in","t"}`` or a starting five / reset ``{"team","players","t"}``."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)

    team: Team
    t: float | None = Field(None, ge=0)
    out: str | None = Field(None, pattern=_ID)
    in_: str | None = Field(None, alias="in", pattern=_ID)
    players: list[Annotated[str, Field(pattern=_ID)]] | None = Field(None, min_length=1, max_length=5)

    @model_validator(mode="after")
    def _shape(self) -> LineupIn:
        sub = self.out is not None or self.in_ is not None
        if sub == (self.players is not None):
            raise ValueError("send either players (lineup) or out + in (substitution)")
        if sub and (self.out is None or self.in_ is None or self.out == self.in_):
            raise ValueError("a substitution needs distinct out and in player ids")
        if self.players is not None and len(set(self.players)) != len(self.players):
            raise ValueError("duplicate player in lineup")
        return self

    def player_ids(self) -> list[str]:
        return list(self.players) if self.players is not None else [self.out or "", self.in_ or ""]

    def to_controller(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


class ScoreboardRoiIn(BaseModel):
    """Venue scoreboard ROI: 4 corners (TL, TR, BR, BL) and named digit cells [x1,y1,x2,y2]."""

    model_config = _STRICT

    corners: list[Point] = Field(min_length=4, max_length=4)
    cells: dict[Annotated[str, Field(pattern=_CELL)], Box] = Field(min_length=1, max_length=64)
    space: Literal["image"] = "image"  # cell boxes are source-image pixels
    image_size: ImageSize | None = None

    @model_validator(mode="after")
    def _geometry(self) -> ScoreboardRoiIn:
        _check_quad(self.corners)
        for name, (x1, y1, x2, y2) in self.cells.items():
            if not (x2 > x1 and y2 > y1):
                raise ValueError(f"cell {name!r} needs x2 > x1 and y2 > y1")
            if not _inside_convex(self.corners, ((x1 + x2) / 2, (y1 + y2) / 2)):
                raise ValueError(f"cell {name!r} is outside the scoreboard corners")
        if self.image_size is not None:
            w, h = self.image_size
            pts = [*self.corners, *([b[0], b[1]] for b in self.cells.values()),
                   *([b[2], b[3]] for b in self.cells.values())]
            if any(not (0 <= u <= w and 0 <= v <= h) for u, v in pts):
                raise ValueError("corners and cells must lie inside image_size")
        return self


class CalibrationIn(BaseModel):
    """Image points with matching court points (ft) or landmark names (``meta.landmarks``)."""

    model_config = _STRICT

    image_points: list[Point] = Field(min_length=4, max_length=64)
    court_points: list[Point] | None = Field(None, min_length=4, max_length=64)
    landmarks: list[Annotated[str, Field(pattern=_LANDMARK)]] | None = Field(None, min_length=4, max_length=64)
    ransac_thresh_ft: float | None = Field(None, gt=0, le=10)
    # Pixel size of the frame the points were clicked on (the /api/frame.jpg the dashboard
    # loaded), so the controller can rescale if that JPEG is not the full source resolution.
    image_size: ImageSize | None = None

    @model_validator(mode="after")
    def _pairs(self) -> CalibrationIn:
        targets = [v for v in (self.court_points, self.landmarks) if v is not None]
        if len(targets) != 1:
            raise ValueError("send exactly one of court_points or landmarks")
        if len(targets[0]) != len(self.image_points):
            raise ValueError("image_points and court_points/landmarks must have the same length")
        if self.landmarks is not None and len(set(self.landmarks)) != len(self.landmarks):
            raise ValueError("duplicate landmark")
        if self.image_size is not None:
            w, h = self.image_size
            if any(not (0 <= u <= w and 0 <= v <= h) for u, v in self.image_points):
                raise ValueError("image_points must lie inside image_size")
        return self

    def to_controller(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class ControlIn(BaseModel):
    """Playback / narration control. ``mute``/``unmute``/``dead_ball`` extend the contract."""

    model_config = _STRICT

    action: Literal["pause", "resume", "speed", "mute", "unmute", "dead_ball"]
    value: float | None = None

    def to_controller(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def _polygon_area(pts: list[list[float]]) -> float:
    """Shoelace area; positive for TL, TR, BR, BL order in image coordinates (v down)."""
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1], strict=True))


def _cross(o: list[float], a: list[float], b: list[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _check_quad(corners: list[list[float]]) -> None:
    """Corners must form a convex quad in TL, TR, BR, BL order, else rectification mirrors or
    rotates the scoreboard and the digit reader silently fails."""
    if _polygon_area(corners) < 4.0:
        if abs(_polygon_area(corners)) >= 4.0:
            raise ValueError("scoreboard corners must be in order top-left, top-right, bottom-right, bottom-left")
        raise ValueError("scoreboard corners are degenerate")
    n = len(corners)
    if any(_cross(corners[i], corners[(i + 1) % n], corners[(i + 2) % n]) <= 0 for i in range(n)):
        raise ValueError("scoreboard corners must form a convex quadrilateral")
    tl, tr, br, bl = corners
    if not (tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]):
        raise ValueError("scoreboard corners must be in order top-left, top-right, bottom-right, bottom-left")


def _inside_convex(quad: list[list[float]], p: tuple[float, float]) -> bool:
    n = len(quad)
    return all(_cross(quad[i], quad[(i + 1) % n], [p[0], p[1]]) >= 0 for i in range(n))


# --------------------------------------------------------------------------- security helpers


def _same_origin(origin: str, host: str | None, allowed: list[str]) -> bool:
    if origin in allowed:
        return True
    try:
        netloc = urlsplit(origin).netloc
    except ValueError:
        return False
    return bool(host) and netloc.lower() == str(host).lower()


_PRIVATE_SUFFIXES = (".localhost", ".local", ".lan", ".home.arpa", ".internal")


def _host_name(host: str) -> str:
    """Host header without its port; IPv6 brackets removed."""
    host = host.strip().lower()
    if host.startswith("["):
        return host[1:].split("]", 1)[0]
    name, sep, port = host.rpartition(":")
    return name if sep and port.isdigit() else host


def _host_ok(host: str | None, allowed: list[str]) -> bool:
    if not host or "*" in allowed:
        return True  # non-browser clients may omit Host; browsers always send it
    name = _host_name(host).rstrip(".")
    try:
        ipaddress.ip_address(name)
        return True  # an IP literal cannot be a rebinding target
    except ValueError:
        pass
    if "." not in name or name.endswith(_PRIVATE_SUFFIXES):
        return True
    for entry in allowed:
        entry = entry.strip().lower()
        if name == entry or (entry.startswith("*.") and name.endswith(entry[1:])):
            return True
    return False


def _token_ok(expected: str | None, given: str | None) -> bool:
    if expected is None:
        return True
    if not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" else None


class _GuardMiddleware:
    """Host allowlist, body-size cap, foreign-Origin refusal on unsafe methods, security headers."""

    _UNSAFE = frozenset({"POST", "PUT", "PATCH", "DELETE"})

    def __init__(self, app: ASGIApp, max_body: int, allowed_origins: list[str],
                 allowed_hosts: list[str] | None = None, upload_max_body: int | None = None) -> None:
        self.app = app
        self.max_body = max_body
        self.allowed = list(allowed_origins)
        self.allowed_hosts = list(allowed_hosts or [])
        self.upload_max_body = upload_max_body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        refusal = self._refusal(scope, headers)
        if refusal is not None:
            status, detail = refusal
            await JSONResponse({"detail": detail}, status_code=status)(scope, receive, send)
            return
        host = headers.get("host", "")
        csp = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
            f"connect-src 'self' ws://{host} wss://{host}; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'self'"
        )

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                extra = [
                    (b"x-content-type-options", b"nosniff"),
                    (b"referrer-policy", b"no-referrer"),
                    (b"x-frame-options", b"DENY"),
                    (b"content-security-policy", csp.encode("latin-1")),
                ]
                message = {**message, "headers": list(message.get("headers", [])) + extra}
            await send(message)

        await self.app(scope, self._capped(receive, self._body_limit(scope)), send_with_headers)

    def _body_limit(self, scope: Scope) -> int:
        if scope.get("path") == "/api/studio/upload" and self.upload_max_body is not None:
            return self.upload_max_body
        return self.max_body

    def _refusal(self, scope: Scope, headers: dict[str, str]) -> tuple[int, str] | None:
        if not _host_ok(headers.get("host"), self.allowed_hosts):
            return 400, "host not allowed"
        if scope.get("method") in self._UNSAFE:
            origin = headers.get("origin")
            if origin and not _same_origin(origin, headers.get("host"), self.allowed):
                return 403, "cross-origin request refused"
        try:
            length = int(headers.get("content-length", "0") or 0)
        except ValueError:
            return 400, "bad content-length"
        if length > self._body_limit(scope):
            return 413, "request body too large"
        return None

    def _capped(self, receive: Receive, limit: int | None = None) -> Receive:
        seen = 0

        async def capped() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > (self.max_body if limit is None else limit):
                    raise HTTPException(413, "request body too large")
            return message

        return capped


# --------------------------------------------------------------------------- app factory


def create_app(
    hub: Hub,
    controller: Controller | None = None,
    auth_token: str | None = None,
    config: ServerConfig | None = None,
    *, upload_max_body: int | None = None,
) -> FastAPI:
    """Build the CourtVision FastAPI app (see module docstring)."""
    cfg = config or ServerConfig()
    token = auth_token or None
    static_dir = Path(cfg.static_dir) if cfg.static_dir else STATIC_DIR

    @asynccontextmanager
    async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
        hub.bind_loop(asyncio.get_running_loop())
        yield
        hub.close_clients()

    extra: dict[str, Any] = {}
    if not cfg.telemetry and "telemetry" in inspect.signature(FastAPI.__init__).parameters:
        extra["telemetry"] = {"tracing": False, "metrics": False, "logs": False}
    app = FastAPI(
        title="CourtVision",
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json" if cfg.openapi else None,
        default_response_class=HubJSONResponse,
        **extra,
    )
    app.state.hub = hub
    app.state.controller = controller
    app.state.started = time.monotonic()
    app.add_middleware(_GuardMiddleware, max_body=cfg.max_body_bytes, allowed_origins=cfg.allowed_origins,
                       allowed_hosts=cfg.allowed_hosts, upload_max_body=upload_max_body)
    _install_error_handlers(app)

    def require_write(request: Request) -> None:
        if not _token_ok(token, _bearer(request)):
            raise HTTPException(401, "missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})

    def require_read(request: Request) -> None:
        if cfg.protect_reads:
            require_write(request)

    app.state.require_read = require_read
    app.state.require_write = require_write

    _register_reads(app, hub, cfg, controller, Depends(require_read))
    _register_writes(app, hub, cfg, controller, Depends(require_write))
    from courtvision.server.ask import Asker, register_ask

    app.state.asker = Asker(hub)
    register_ask(app, app.state.asker, Depends(require_read), Depends(require_write))
    _register_ws(app, hub, cfg, token)
    _register_static(app, static_dir)
    return app


def _install_error_handlers(app: FastAPI) -> None:
    async def validation_error(_: Request, exc: Exception) -> Response:
        errors = exc.errors() if isinstance(exc, RequestValidationError) else [str(exc)]
        return HubJSONResponse({"detail": to_jsonable(_clean_errors(errors))}, status_code=400)

    app.add_exception_handler(RequestValidationError, validation_error)


def _clean_errors(errors: Any) -> Any:
    """Drop echoed inputs and exception objects from pydantic errors (keep loc/msg/type)."""
    if not isinstance(errors, (list, tuple)):
        return errors
    out = []
    for e in errors:
        if isinstance(e, Mapping):
            out.append({k: e[k] for k in ("loc", "msg", "type") if k in e})
        else:
            out.append(str(e))
    return out


async def _invoke(controller: Any, method: str, *args: Any) -> Any:
    """Call a controller method; maps missing controller -> 503, bad input -> 400."""
    if controller is None:
        raise HTTPException(503, "no pipeline controller attached")
    fn: Callable[..., Any] | None = getattr(controller, method, None)
    if fn is None:
        raise HTTPException(501, f"controller does not implement {method}")
    try:
        result = fn(*args)
        if inspect.isawaitable(result):
            result = await result
    except HTTPException:
        raise
    except (ValueError, LookupError) as exc:
        raise HTTPException(400, str(exc) or type(exc).__name__) from exc
    except NotImplementedError as exc:
        raise HTTPException(501, f"{method} not implemented") from exc
    except Exception as exc:
        log.exception("controller.%s failed", method)
        raise HTTPException(500, f"controller error in {method}") from exc
    return result


def _json(content: Any, status_code: int = 200) -> Response:
    return HubJSONResponse(content, status_code=status_code)


def _ok(result: Any) -> Response:
    return _json({"ok": True, "result": result})


def _roster_teams(hub: Hub) -> dict[str, str] | None:
    meta = hub.latest("meta")
    if not meta or not meta.get("roster"):
        return None
    return {str(e.get("player_id")): str(e.get("team")) for e in meta["roster"] if isinstance(e, Mapping)}


# --------------------------------------------------------------------------- routes


def _register_reads(app: FastAPI, hub: Hub, cfg: ServerConfig, controller: Any, auth: Any) -> None:
    @app.get("/healthz", response_class=HubJSONResponse)
    async def healthz() -> Response:
        return _json({"ok": True, "controller": controller is not None, "clients": hub.client_count,
                      "uptime_s": round(time.monotonic() - app.state.started, 1)})

    @app.get("/api/state", dependencies=[auth], response_class=HubJSONResponse)
    async def state() -> Response:
        frame = hub.latest("frame")
        stats = hub.latest("stats")
        return _json({
            "t": frame.get("t") if frame else None,
            "meta": hub.latest("meta"),
            "frame": frame,
            "status": hub.latest("status"),
            "stats_t": stats.get("t") if stats else None,
            "controller": controller is not None,
            "hub": hub.info(),
        })

    @app.get("/api/events", dependencies=[auth], response_class=HubJSONResponse)
    async def events(
        kind: Annotated[list[EventKind] | None, Query()] = None,
        since: Annotated[float | None, Query(allow_inf_nan=False)] = None,
        limit: Annotated[int | None, Query(ge=1)] = None,
    ) -> Response:
        n = min(limit or cfg.events_limit_default, cfg.events_limit_max)
        kinds = [k.value for k in kind] if kind else None
        return _json({"events": hub.events(kinds=kinds, since=since, limit=n)})

    @app.get("/api/boxscore", dependencies=[auth], response_class=HubJSONResponse)
    async def boxscore() -> Response:
        stats = hub.latest("stats")
        return _json({"t": stats.get("t") if stats else None, "stats": stats.get("stats") if stats else None})

    @app.get("/api/transcript", dependencies=[auth], response_class=HubJSONResponse)
    async def transcript(limit: Annotated[int | None, Query(ge=1, le=5000)] = None) -> Response:
        return _json({"utterances": hub.transcript(limit), "captions": hub.captions()})

    @app.get("/api/insights", dependencies=[auth], response_class=HubJSONResponse)
    async def insights() -> Response:
        return _json({"insights": hub.insights()})

    @app.get("/api/frame.jpg", dependencies=[auth], response_class=Response)
    async def frame_jpeg() -> Response:
        data = await _invoke(controller, "latest_frame_jpeg")
        if data is None or len(data) == 0:  # len(): a cv2.imencode ndarray has no truth value
            raise HTTPException(404, "no frame available yet")
        return Response(bytes(data), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def _register_writes(app: FastAPI, hub: Hub, cfg: ServerConfig, controller: Any, auth: Any) -> None:
    @app.post("/api/events", dependencies=[auth], response_class=HubJSONResponse)
    async def add_event(body: ManualEventIn) -> Response:
        payload = body.to_controller()
        if body.status is not EventStatus.RETRACTED:
            payload["team"] = _check_event(hub, body)
        return _ok(await _invoke(controller, "add_manual", payload))

    @app.post("/api/scoreboard", dependencies=[auth], response_class=HubJSONResponse)
    async def set_scoreboard(body: ScoreboardIn) -> Response:
        return _ok(await _invoke(controller, "set_scoreboard", body.to_controller()))

    @app.post("/api/lineup", dependencies=[auth], response_class=HubJSONResponse)
    async def set_lineup(body: LineupIn) -> Response:
        _check_players(hub, body.team, body.player_ids(), require_known=True)
        return _ok(await _invoke(controller, "set_lineup", body.to_controller()))

    @app.post("/api/scoreboard_roi", dependencies=[auth], response_class=HubJSONResponse)
    async def set_scoreboard_roi(body: ScoreboardRoiIn) -> Response:
        roi = body.model_dump(mode="json", exclude_none=True)
        return _ok(await _invoke(controller, "set_scoreboard_roi", roi))

    @app.post("/api/calibration", dependencies=[auth], response_class=HubJSONResponse)
    async def calibrate(body: CalibrationIn) -> Response:
        if body.landmarks is not None:
            known = (hub.latest("meta") or {}).get("landmarks") or {}
            unknown = [n for n in body.landmarks if known and n not in known]
            if unknown:
                raise HTTPException(400, f"unknown landmark(s): {', '.join(unknown)}")
        return _ok(await _invoke(controller, "calibrate", body.to_controller()))

    @app.post("/api/control", dependencies=[auth], response_class=HubJSONResponse)
    async def control(body: ControlIn) -> Response:
        if body.action == "speed":
            v = body.value
            if v is None or not math.isfinite(v) or not 0 < v <= cfg.max_speed:
                raise HTTPException(400, f"speed needs 0 < value <= {cfg.max_speed}")
        return _ok(await _invoke(controller, "control", body.to_controller()))


def _team_of(pid: str | None, roster_teams: dict[str, str] | None) -> str | None:
    """Team of a player id: the ``meta`` roster first, then the ``home-`` / ``away-`` id prefix
    (types.py provisional ids; same rule as stats ``TeamResolver``)."""
    if pid is None:
        return None
    if roster_teams is not None and pid in roster_teams:
        team = roster_teams[pid]
        return team if team in (Team.HOME.value, Team.AWAY.value) else None
    prefix = pid.split("-", 1)[0].lower()
    return prefix if prefix in (Team.HOME.value, Team.AWAY.value) else None


def _check_players(hub: Hub, team: Team | None, ids: list[str | None], require_known: bool) -> None:
    """Every id must be on ``team`` (roster or id prefix). With a roster in ``meta``, lineups also
    reject unknown ids; events may name provisional track ids such as ``home-t12``."""
    teams = _roster_teams(hub)
    for pid in ids:
        if pid is None:
            continue
        if require_known and teams is not None and pid not in teams:
            raise HTTPException(400, f"player {pid!r} is not on the roster")
        actual = _team_of(pid, teams)
        if team is not None and actual is not None and actual != team.value:
            raise HTTPException(400, f"player {pid!r} is not on the {team.value} roster")


def _check_event(hub: Hub, body: ManualEventIn) -> str | None:
    """Team consistency for a confirmed manual event; returns the event's team (filled in from
    the actor when omitted, since ``Event.team`` is always the actor's team)."""
    teams = _roster_teams(hub)
    _check_players(hub, body.team, [body.player_id], require_known=False)
    team = body.team.value if body.team is not None else _team_of(body.player_id, teams)
    if team is None and body.kind not in _TEAMLESS_OK:
        raise HTTPException(400, f"{body.kind.value} needs a team (or a player_id whose team is known)")
    other = _team_of(body.other_player_id, teams)
    if team is not None and other is not None:
        if body.kind in _SAME_TEAM_PAIR and other != team:
            raise HTTPException(400, f"{body.kind.value}: other_player_id must be a teammate")
        if body.kind in _OPPONENT_PAIR and other == team:
            raise HTTPException(400, f"{body.kind.value}: other_player_id must be an opponent")
    return team


def _register_ws(app: FastAPI, hub: Hub, cfg: ServerConfig, token: str | None) -> None:
    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host")
        if (
            not _host_ok(host, cfg.allowed_hosts)
            or not _token_ok(token, websocket.query_params.get("token"))
            or (origin and not _same_origin(origin, host, cfg.allowed_origins))
        ):
            await websocket.close(code=1008)  # policy violation; before accept -> HTTP 403
            return
        await websocket.accept()
        client = hub.subscribe()
        try:
            await _serve_client(websocket, client, hub, cfg)
        finally:
            hub.unsubscribe(client)


async def _serve_client(websocket: WebSocket, client: HubClient, hub: Hub, cfg: ServerConfig) -> None:
    pump = asyncio.create_task(_pump(websocket, client, hub))
    reader = asyncio.create_task(_read(websocket, client, cfg))
    try:
        await asyncio.wait({pump, reader}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (pump, reader):
            task.cancel()
        for task in (pump, reader):
            with suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError, OSError):
                await task


async def _pump(websocket: WebSocket, client: HubClient, hub: Hub) -> None:
    while True:
        text = await client.get()
        if text is None:  # hub shut down
            with suppress(RuntimeError, OSError):
                await websocket.close(code=1001)
            return
        await websocket.send_text(text)
        if client.overflowed and hub.config.disconnect_on_overflow:
            log.warning("closing slow WebSocket client %d for resync", client.id)
            with suppress(RuntimeError, OSError):
                await websocket.close(code=1013)  # try again later: reconnect gets a clean replay
            return


async def _read(websocket: WebSocket, client: HubClient, cfg: ServerConfig) -> None:
    while True:
        message = await websocket.receive()
        if message.get("type") == "websocket.disconnect":
            return
        text = message.get("text")
        if not text or len(text) > cfg.ws_receive_max_bytes:
            continue  # binary frames and oversized messages are ignored
        reply = _ping_reply(text)
        if reply is not None:
            client.offer("pong", reply)


def _ping_reply(text: str) -> str | None:
    try:
        msg = json.loads(text)
    except ValueError:
        return None
    if not isinstance(msg, dict) or msg.get("type") != "ping":
        return None
    ping_id = msg.get("id")
    if not isinstance(ping_id, (int, str)) or isinstance(ping_id, bool):
        ping_id = None
    return dumps({"type": "pong", "t": None, "id": ping_id, "server_time": time.time()})


def _register_static(app: FastAPI, static_dir: Path) -> None:
    index = static_dir / "index.html"

    @app.get("/", include_in_schema=False)
    async def dashboard() -> Response:
        if not index.is_file():
            raise HTTPException(404, "dashboard not installed")
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    else:
        log.warning("static directory %s missing; dashboard disabled", static_dir)

