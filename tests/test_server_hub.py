"""Hub: JSON encoding, replay, drop-oldest queues, frame throttling, thread-safe publishing."""

from __future__ import annotations

import asyncio
import enum
import json
import math
import threading
from dataclasses import dataclass, field

import numpy as np
import pytest

from courtvision.server.hub import (
    Hub,
    HubConfig,
    HubJSONEncoder,
    caption_msg,
    dumps,
    event_msg,
    frame_msg,
    insight_msg,
    meta_msg,
    stats_msg,
    to_jsonable,
    utterance_msg,
)
from courtvision.types import (
    BallState,
    BBox,
    Event,
    EventKind,
    EventStatus,
    FrameState,
    Insight,
    PlayerState,
    Roster,
    RosterEntry,
    ScoreboardReading,
    Team,
    Utterance,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@dataclass(frozen=True)
class FakeCourt:
    league: str = "nba"
    length: float = 94.0
    width: float = 50.0
    three_pt_radius: float = 23.75

    def landmarks(self) -> dict[str, tuple[float, float]]:
        return {"center": (47.0, 25.0), "left_baseline_bottom": (0.0, 0.0)}


@dataclass
class FakeStats:
    t: float
    score: dict[Team, int] = field(default_factory=lambda: {Team.HOME: 3, Team.AWAY: 0})

    def to_dict(self) -> dict:
        return {"t": self.t, "score": {k.value: v for k, v in self.score.items()}, "efg": float("nan")}


class Color(enum.IntEnum):
    RED = 1


def ev(i: int, kind: EventKind = EventKind.PASS, revision: int = 0, t: float | None = None, **kw) -> Event:
    return Event(f"e{i:06d}", kind, float(i) if t is None else t, i * 30, revision=revision, **kw)


def drain(client) -> list[dict]:
    out = []
    while (text := client.get_nowait()) is not None:
        out.append(json.loads(text))
    return out


# --------------------------------------------------------------------------- encoding


def test_to_jsonable_handles_numpy_enums_dataclasses_and_nan():
    payload = {
        Team.HOME: np.float32(1.5),
        "i": np.int64(7),
        "b": np.bool_(True),
        "arr": np.array([[1.0, np.nan], [np.inf, 2.0]]),
        "nan": float("nan"),
        "ninf": -math.inf,
        "enum": EventStatus.CONFIRMED,
        "intenum": Color.RED,
        "tuple": (1, 2),
        "set": {3, 1, 2},
        "bbox": BBox(1, 2, 3, 4),
        "event": ev(1, team=Team.AWAY),
        3: "int key",
    }
    out = to_jsonable(payload)
    assert out["home"] == 1.5 and isinstance(out["home"], float)
    assert out["i"] == 7 and type(out["i"]) is int
    assert out["b"] is True
    assert out["arr"] == [[1.0, None], [None, 2.0]]
    assert out["nan"] is None and out["ninf"] is None
    assert out["enum"] == "confirmed" and out["intenum"] == 1
    assert out["tuple"] == [1, 2] and out["set"] == [1, 2, 3]
    assert out["bbox"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    assert out["event"]["kind"] == "pass" and out["event"]["team"] == "away"  # Event.to_dict used
    assert out["3"] == "int key"


def test_dumps_is_strict_json_without_nan_tokens():
    text = dumps({"a": float("nan"), "b": [np.float64("inf")], "c": Team.AWAY})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None], "c": "away"}


def test_hub_json_encoder_class():
    text = json.dumps({"x": np.float64("nan"), "y": np.arange(3), "t": Team.HOME}, cls=HubJSONEncoder)
    assert json.loads(text) == {"x": None, "y": [0, 1, 2], "t": "home"}


def test_message_builders_follow_ws_schema():
    roster = Roster([RosterEntry("home-23", Team.HOME, "23", "Ava Stone")], {Team.HOME: "Hawks", Team.AWAY: "Owls"})
    frame = FrameState(
        frame_idx=12,
        t=0.4,
        players=[PlayerState("home-23", Team.HOME, 10.0, 20.0, vx=3.0, vy=4.0),
                 PlayerState("away-t7", Team.AWAY, 30.0, 25.0, jersey="7")],
        ball=BallState(visible=True, court_xy=(11.0, 21.0), z=np.float64(6.5)),
        scoreboard=ScoreboardReading(t=0.4, period=2, game_clock=431.25, shot_clock=float("nan")),
    )
    msg = json.loads(dumps(frame_msg(frame, roster, holder_id="home-23", possession=Team.HOME)))
    assert msg["type"] == "frame" and msg["t"] == 0.4 and msg["frame_idx"] == 12
    p0, p1 = msg["players"]
    assert p0 == {"id": "home-23", "team": "home", "jersey": "23", "name": "Ava Stone", "x": 10.0, "y": 20.0, "speed": 5.0}
    assert p1["name"] is None and p1["jersey"] == "7"
    assert msg["ball"] == {"x": 11.0, "y": 21.0, "z": 6.5, "visible": True, "holder": "home-23"}
    assert msg["possession"] == "home"
    assert msg["clock"] == {"period": 2, "game": 431.2, "shot": None}

    meta = json.loads(dumps(meta_msg(roster, FakeCourt(), rules={"n_periods": 4}, colors={"home": "#123456"})))
    assert meta["type"] == "meta" and meta["teams"] == {"home": "Hawks", "away": "Owls"}
    assert meta["roster"][0]["team"] == "home" and meta["court"]["length"] == 94.0
    assert meta["landmarks"]["center"] == [47.0, 25.0] and meta["rules"] == {"n_periods": 4}
    assert meta["colors"] == {"home": "#123456"}

    st = json.loads(dumps(stats_msg(FakeStats(t=9.0))))
    assert st == {"type": "stats", "t": 9.0, "stats": {"t": 9.0, "score": {"home": 3, "away": 0}, "efg": None}}
    assert event_msg(ev(3))["event"]["event_id"] == "e000003"
    u = utterance_msg(Utterance("u0001", "Nice pass.", "play_by_play", 2.0))
    assert u["type"] == "utterance" and u["utterance"]["text"] == "Nice pass."
    i = insight_msg(Insight("i0001", 3.0, "run", "Owls 9-0 run", team=Team.AWAY, evidence={"n_poss": 6}))
    assert json.loads(dumps(i))["insight"]["team"] == "away"
    assert caption_msg(4.0, 120.5, "Shot, left wing.", "ad")["channel"] == "ad"


# --------------------------------------------------------------------------- replay


async def test_replay_has_latest_snapshots_and_last_200_history_items():
    hub = Hub()
    await hub.publish({"type": "meta", "t": 0.0, "teams": {"home": "H", "away": "A"}})
    await hub.publish({"type": "stats", "t": 1.0, "stats": {"v": 1}})
    await hub.publish({"type": "stats", "t": 2.0, "stats": {"v": 2}})
    await hub.publish({"type": "frame", "t": 1.0, "frame_idx": 30})
    await hub.publish({"type": "frame", "t": 2.0, "frame_idx": 60})
    for i in range(250):
        await hub.publish(ev(i))
    await hub.publish(ev(10_000, EventKind.DRIBBLE))  # never replayed
    await hub.publish(ev(249, EventKind.PASS, revision=2, attrs={"completed": False}))
    for i in range(205):
        await hub.publish(Utterance(f"u{i:04d}", f"line {i}", "play_by_play", float(i)))
    await hub.publish(Insight("i0001", 1.0, "run", "first"))
    await hub.publish(Insight("i0001", 2.0, "run", "updated"))
    await hub.publish(caption_msg(3.0, 3.0, "cue"))

    client = hub.subscribe()
    msgs = drain(client)
    assert msgs[0]["type"] == "meta"
    assert msgs[1] == {"type": "stats", "t": 2.0, "stats": {"v": 2}}
    assert msgs[-1]["type"] == "frame" and msgs[-1]["frame_idx"] == 60
    events = [m["event"] for m in msgs if m["type"] == "event"]
    assert len(events) == 200
    assert all(e["kind"] != "dribble" for e in events)
    assert len({e["event_id"] for e in events}) == 200
    rev = [e for e in events if e["event_id"] == "e000249"]
    assert rev[0]["revision"] == 2 and rev[0]["attrs"] == {"completed": False}
    utts = [m["utterance"] for m in msgs if m["type"] == "utterance"]
    assert len(utts) == 200 and utts[-1]["text"] == "line 204"
    insights = [m["insight"] for m in msgs if m["type"] == "insight"]
    assert [i["text"] for i in insights] == ["updated"]
    assert sum(m["type"] == "caption" for m in msgs) == 1
    seqs = [m["type"] for m in msgs[2:-1]]
    assert set(seqs) == {"event", "utterance", "insight", "caption"}
    # REST log keeps everything, dribbles included, latest revision per id
    assert len(hub.events()) == 251
    assert hub.events(kinds=["dribble"])[0]["event_id"] == "e010000"


async def test_stale_revision_is_ignored_and_not_broadcast():
    hub = Hub()
    client = hub.subscribe()
    await hub.publish(ev(1, revision=3, status=EventStatus.CONFIRMED))
    await hub.publish(ev(1, revision=1))
    msgs = drain(client)
    assert len(msgs) == 1 and msgs[0]["event"]["revision"] == 3
    assert hub.events()[0]["status"] == "confirmed"


async def test_rest_helpers_filter_events_and_cap_transcript():
    hub = Hub(HubConfig(transcript_max=3))
    for i in range(5):
        await hub.publish(ev(i, EventKind.SHOT_MADE if i % 2 else EventKind.PASS, t=float(i)))
        await hub.publish(Utterance(f"u{i}", f"l{i}", "color", float(i)))
    assert [e["event_id"] for e in hub.events(kinds=["shot_made"])] == ["e000001", "e000003"]
    assert [e["event_id"] for e in hub.events(since=3.0)] == ["e000003", "e000004"]
    assert [e["event_id"] for e in hub.events(limit=2)] == ["e000003", "e000004"]
    assert [u["text"] for u in hub.transcript()] == ["l2", "l3", "l4"]
    assert hub.latest("frame") is None


async def test_publish_validates_messages():
    hub = Hub()
    with pytest.raises(ValueError):
        await hub.publish({"t": 1.0})
    with pytest.raises(TypeError):
        await hub.publish(["not", "a", "mapping"])  # type: ignore[arg-type]
    await hub.publish({"type": "custom"})  # unknown types are broadcast, t defaults to null
    client = hub.subscribe()
    assert drain(client) == []  # ...but not replayed
    assert hub.published["custom"] == 1


# --------------------------------------------------------------------------- queues


async def test_drop_oldest_on_overflow():
    hub = Hub(HubConfig(queue_size=3, max_rate_hz={}))
    client = hub.subscribe()
    for i in range(5):
        await hub.publish(Utterance(f"u{i}", f"line {i}", "play_by_play", float(i)))
    assert [m["utterance"]["text"] for m in drain(client)] == ["line 2", "line 3", "line 4"]
    assert client.dropped == 2 and client.overflowed


async def test_superseded_snapshots_are_dropped_before_events():
    hub = Hub(HubConfig(queue_size=4, max_rate_hz={}))
    client = hub.subscribe()
    for i in range(3):
        await hub.publish({"type": "frame", "t": float(i), "frame_idx": i})
    for i in range(3):
        await hub.publish(ev(i))
    msgs = drain(client)
    assert [m["type"] for m in msgs] == ["frame", "event", "event", "event"]
    assert msgs[0]["frame_idx"] == 2  # the newest frame survived
    assert client.dropped == 2 and not client.overflowed


async def test_frame_messages_are_throttled_to_15hz_latest_wins():
    clock = FakeClock()
    hub = Hub(clock=clock)
    client = hub.subscribe()
    for i in range(10):
        await hub.publish({"type": "frame", "t": i / 30, "frame_idx": i})
    await hub.publish(ev(1))  # non-snapshot types are never throttled
    msgs = drain(client)
    assert [m.get("frame_idx") for m in msgs if m["type"] == "frame"] == [0]
    assert [m["type"] for m in msgs] == ["frame", "event"]
    clock.now += 1 / 15
    await asyncio.sleep(1 / 15 + 0.05)  # the coalescing timer fires on the loop
    msgs = drain(client)
    assert [m["frame_idx"] for m in msgs] == [9]
    clock.now += 1.0
    await hub.publish({"type": "frame", "t": 2.0, "frame_idx": 60})
    assert [m["frame_idx"] for m in drain(client)] == [60]


async def test_rate_is_per_client():
    clock = FakeClock()
    hub = Hub(clock=clock)
    a = hub.subscribe()
    await hub.publish({"type": "frame", "t": 0.0, "frame_idx": 0})
    b = hub.subscribe()  # its replay frame counts against its own budget
    await hub.publish({"type": "frame", "t": 0.01, "frame_idx": 1})
    assert [m["frame_idx"] for m in drain(a)] == [0]
    assert [m["frame_idx"] for m in drain(b)] == [0]
    a.flush_pending()
    b.flush_pending()
    assert [m["frame_idx"] for m in drain(a)] == [1]
    assert [m["frame_idx"] for m in drain(b)] == [1]


async def test_publish_threadsafe_from_another_thread():
    hub = Hub(HubConfig(queue_size=1000))
    hub.bind_loop()
    client = hub.subscribe()

    def produce() -> None:
        for i in range(50):
            hub.publish_threadsafe(ev(i, attrs={"xfg": np.float32(0.25)}))

    worker = threading.Thread(target=produce)
    worker.start()
    received = []
    async with asyncio.timeout(5):
        while len(received) < 50:
            received.append(json.loads(await client.get()))
    worker.join()
    assert [m["event"]["event_id"] for m in received] == [f"e{i:06d}" for i in range(50)]
    assert received[0]["event"]["attrs"]["xfg"] == 0.25


def test_publish_threadsafe_without_a_loop_records_state_for_replay():
    hub = Hub()
    hub.publish_threadsafe({"type": "meta", "t": 0.0, "roster": []})
    hub.publish_threadsafe(ev(1))
    assert [m["type"] for m in hub.replay_messages()] == ["meta", "event"]


async def test_close_clients_ends_consumers():
    hub = Hub()
    client = hub.subscribe()
    waiter = asyncio.create_task(client.get())
    await asyncio.sleep(0)
    hub.close_clients()
    async with asyncio.timeout(1):
        assert await waiter is None
    assert hub.client_count == 0
    await hub.publish(ev(1))  # no clients left: just recorded
    assert client.get_nowait() is None


DASHBOARD_COURT_FIELDS = ("length", "width", "rim_radius", "basket_offset", "backboard_offset", "three_pt_radius",
                          "corner_three_dist", "lane_width", "free_throw_dist", "restricted_radius", "center_circle_radius")


@pytest.mark.parametrize("league", ["nba", "fiba", "nfhs"])
def test_meta_msg_with_the_real_courtspec(league: str):
    try:
        from courtvision.geometry.court import court_for
    except Exception as exc:  # geometry is built in parallel; skip until it imports
        pytest.skip(f"geometry not importable yet: {exc}")
    msg = json.loads(dumps(meta_msg(None, court_for(league))))
    for key in DASHBOARD_COURT_FIELDS:  # the dashboard draws the court from these fields
        assert isinstance(msg["court"][key], (int, float)), key
    assert len(msg["landmarks"]) >= 4
    assert all(len(xy) == 2 for xy in msg["landmarks"].values())
