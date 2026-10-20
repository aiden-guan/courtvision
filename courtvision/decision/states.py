"""Jev state builders for B1-B5 (jev_integration_design.md §6) plus the bucket helpers.

Rules every builder follows (§6.0, D16, RESEARCH.md §7A.7):

* Numbers become named buckets or words in Python; Jev never counts or compares.
* Pseudonymised: roles ("the shooter", "Home's ball handler") or per-game tokens (``{P07}``) only,
  never names or player ids. Free text passes through ``redact`` with the caller's name list.
* Compact: each state is trimmed to ``StateConfig.max_state_bytes`` (1 KiB by default).
* Local-only data (numbers for the deterministic fallbacks, redaction lists, ids) lives under the
  top-level ``"_local"`` key, which ``client.sanitize`` strips before sending.
"""

from __future__ import annotations

import copy
import enum
import logging
import math
import numbers
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from courtvision.decision.client import sanitize, state_bytes
from courtvision.decision.questions import CARD_KEYS, MAX_GUARD_SENTENCES, MAX_STORYLINES
from courtvision.types import Event, EventKind, EventStatus, Team

log = logging.getLogger(__name__)

LOCAL_KEY = "_local"


def _default_budgets() -> dict[str, int]:
    # Task spec: every state <= 1 KiB. Design §4 allows 1.5 KiB for B3/B4 and asks <= 0.6 KiB for B5.
    return {"B1": 1024, "B2": 1024, "B3": 1024, "B4": 1024, "B5": 614}


@dataclass
class StateConfig:
    """Bucket cut-offs and size limits for Jev states."""

    clutch_window_s: float = 300.0  # NBA.com clutch "Last 5 Minutes" (sourced); Q4/OT scope is our choice (unverified)
    clutch_margin: int = 5  # NBA.com clutch "5 Point Diff or Less" (sourced)
    one_possession_max: int = 3  # design §6.0 margin buckets (design proposal)
    two_possession_max: int = 6  # design §6.0 (design proposal)
    three_possession_max: int = 9  # design §6.0 (design proposal)
    double_digit_max: int = 19  # design §6.0; >= 20 is "comfortably" (design proposal)
    final_minute_s: float = 60.0  # design §6.0 time buckets (design proposal)
    final_two_s: float = 120.0  # design §6.0 (design proposal)
    final_five_s: float = 300.0  # design §6.0 (design proposal)
    late_shot_clock_s: float = 4.0  # design §3 site 4 "shot clock under 4 s (forced shot)" (design proposal)
    mid_shot_clock_s: float = 12.0  # half of a 24 s clock (unverified)
    run_callout_min: int = 8  # narration note: run callouts at >= 8-0 (sourced from narration.md)
    and_one_window_s: float = 1.0  # foul on the shooter within this of the make => and-one (unverified)
    putback_phrase_s: float = 2.0  # "shot right after an offensive rebound" (design §3 site 3 putback window)
    rim_distance_ft: float = 4.0  # "at the rim" (restricted-area radius, NBA 4 ft; sourced)
    short_range_ft: float = 14.0  # pbpstats short mid-range cut-off (unverified here)
    deep_three_ft: float = 28.0  # "deep three" phrase cut-off (unverified)
    max_actions: int = 6  # design §6.1 (design proposal)
    max_recent: int = 5  # design §6.0 (design proposal)
    max_storylines: int = MAX_STORYLINES  # design §6.0
    max_cards: int = len(CARD_KEYS)  # design §5 / RESEARCH §7A.9
    max_card_chars: int = 96  # cards are <= 12 words (design §3 site 6)
    max_state_bytes: dict[str, int] = field(default_factory=_default_budgets)


_DEFAULT_CFG = StateConfig()

# --------------------------------------------------------------------------- number words

_UNITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
          "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINALS = ("zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth")


def number_words(n: int) -> str:
    """0..99 as words (larger values say "many"): states carry no digits outside whitelisted fields."""
    n = int(n)
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS[tens] + ("" if unit == 0 else "-" + _UNITS[unit])
    return "many"


def ordinal_words(n: int) -> str:
    """1 -> "first" ... 10 -> "tenth"; anything else "later"."""
    return _ORDINALS[n] if 0 <= n < len(_ORDINALS) else "later"


# --------------------------------------------------------------------------- situation buckets

TIME_TEMPLATES: dict[str, str] = {
    "under_shot_clock": "less than one shot clock left in {scope}",
    "final_minute": "final minute of {scope}",
    "final_two": "final two minutes of {scope}",
    "final_five": "final five minutes of {scope}",
    "earlier": "earlier in {scope}",
    "unknown": "game clock unknown",
}
MARGIN_TEMPLATES: dict[str, str] = {
    "tied": "the game is tied",
    "one_possession": "{leader} leads; a one-possession game",
    "two_possession": "{leader} leads; a two-possession game",
    "three_possession": "{leader} leads; a three-possession game",
    "double_digits": "{leader} leads by double digits",
    "comfortable": "{leader} leads comfortably",
}
CLUTCH_YES = "yes: late in the game and close"
CLUTCH_NO = "no"
LEAD_CHANGE_TEXT = "the lead just changed hands"
LEAGUE_TEXT = {"nba": "NBA", "fiba": "FIBA", "ncaa_m": "NCAA men", "ncaa_w": "NCAA women", "nfhs": "high school"}


def finite(x: Any) -> float | None:
    """``x`` as a float, or None when it is missing, NaN, infinite or not a number."""
    if x is None or isinstance(x, (bool, np.bool_)):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def time_code(is_last_period_or_ot: bool, clock_s: float | None, shot_clock_full_s: int | None,
              cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Code of the time bucket (keys of ``TIME_TEMPLATES``); "unknown" for a missing/NaN clock."""
    clock_s = finite(clock_s)
    if clock_s is None:
        return "unknown"
    if shot_clock_full_s and clock_s <= shot_clock_full_s:
        return "under_shot_clock"
    if clock_s <= cfg.final_minute_s:
        return "final_minute"
    if clock_s <= cfg.final_two_s:
        return "final_two"
    if clock_s <= cfg.final_five_s:
        return "final_five"
    return "earlier"


def time_bucket(is_last_period_or_ot: bool, clock_s: float | None, shot_clock_full_s: int | None,
                cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Game-clock phrase, e.g. "final two minutes of the game" (design §6.0)."""
    scope = "the game" if is_last_period_or_ot else "the period"
    return TIME_TEMPLATES[time_code(is_last_period_or_ot, clock_s, shot_clock_full_s, cfg)].format(scope=scope)


def margin_code(diff: int, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Code of the margin bucket (keys of ``MARGIN_TEMPLATES``)."""
    d = abs(int(diff))
    if d == 0:
        return "tied"
    if d <= cfg.one_possession_max:
        return "one_possession"
    if d <= cfg.two_possession_max:
        return "two_possession"
    if d <= cfg.three_possession_max:
        return "three_possession"
    if d <= cfg.double_digit_max:
        return "double_digits"
    return "comfortable"


def margin_text(leader: str | None, diff: int, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Score-margin phrase, e.g. "Home leads; a one-possession game" (design §6.0)."""
    return MARGIN_TEMPLATES[margin_code(diff, cfg)].format(leader=leader or "one team")


def is_clutch(is_last_period_or_ot: bool, clock_s: float | None, diff: int, cfg: StateConfig = _DEFAULT_CFG) -> bool:
    """Last period/OT, <= 5:00 left and margin <= 5 (NBA.com clutch filters; scope is config)."""
    clock_s = finite(clock_s)
    return bool(is_last_period_or_ot and clock_s is not None and clock_s <= cfg.clutch_window_s
                and abs(int(diff)) <= cfg.clutch_margin)


def clutch_text(is_last_period_or_ot: bool, clock_s: float | None, diff: int, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """"yes: late in the game and close" or "no" (NBA.com clutch filters; Q4/OT scope is config)."""
    return CLUTCH_YES if is_clutch(is_last_period_or_ot, clock_s, diff, cfg) else CLUTCH_NO


def period_text(period: int | None, n_periods: int = 4, halves: bool = False) -> str:
    """"fourth quarter", "second half", "overtime", "second overtime" (words, no digits)."""
    if period is None or period < 1:
        return "period unknown"
    if period > n_periods:
        k = period - n_periods
        return "overtime" if k == 1 else f"{ordinal_words(k)} overtime"
    return f"{ordinal_words(period)} {'half' if halves else 'quarter'}"


def shot_clock_text(shot_clock_s: float | None, cfg: StateConfig = _DEFAULT_CFG) -> str | None:
    """Shot-clock bucket phrase, or None when there is no (finite) shot clock reading."""
    shot_clock_s = finite(shot_clock_s)
    if shot_clock_s is None:
        return None
    if shot_clock_s <= cfg.late_shot_clock_s:
        return "late in the shot clock"
    if shot_clock_s <= cfg.mid_shot_clock_s:
        return "middle of the shot clock"
    return "early in the shot clock"


def is_run(points: int, opp_points: int, cfg: StateConfig = _DEFAULT_CFG) -> bool:
    """A callout-worthy run: at least ``run_callout_min`` points net (8-0, 10-2, ...; design proposal)."""
    return points >= cfg.run_callout_min and points - opp_points >= cfg.run_callout_min


def run_text(team: str | None, points: int, opp_points: int, cfg: StateConfig = _DEFAULT_CFG) -> str | None:
    """Run callout text ("Home has scored the last eight points"), or None below the callout bar."""
    if not team or not is_run(points, opp_points, cfg):
        return None
    if opp_points == 0:
        return f"{team} has scored the last {number_words(points)} points"
    return f"{team} is on a {number_words(points)} to {number_words(opp_points)} run"


def code_from_text(text: str | None, templates: Mapping[str, str]) -> str | None:
    """Reverse lookup of a bucket phrase (the fallback reads the same vocabulary the builder wrote)."""
    if not text:
        return None
    t = text.lower()
    for code, template in templates.items():
        fixed = max(re.split(r"\{\w+\}", template), key=len).strip(" ;").lower()
        if fixed and fixed in t:
            return code
    return None


# --------------------------------------------------------------------------- event vocabulary

FLAG_PHRASES: dict[str, str] = {
    "dunk": "dunk",
    "alley_oop": "alley-oop",
    "block": "blocked shot",
    "and_one": "and-one chance",
    "drew_contact": "drew contact",
    "buzzer_beater": "beat the buzzer",
    "go_ahead_three_late": "go-ahead three late in the game",
    "steal": "steal",
    "putback": "putback",
}
# Design §3 site 2 highlight rule: dunk, block, and-one, buzzer, lead-taking three late (design proposal).
HIGHLIGHT_FLAGS = frozenset({"dunk", "alley_oop", "block", "and_one", "buzzer_beater", "go_ahead_three_late"})

SHOT_TYPE_PHRASES = {"jumper": "jump shot", "layup": "layup", "dunk": "dunk", "hook": "hook shot",
                     "tip": "tip-in", "free_throw": "free throw"}
TURNOVER_PHRASES = {"steal": "the ball was stolen", "bad_pass": "a bad pass", "out_of_bounds": "out of bounds",
                    "shot_clock": "a shot-clock violation", "travel": "a travel",
                    "offensive_foul": "an offensive foul", "other": "a turnover"}
START_PHRASES = {"jump_ball": "after a jump ball", "inbound": "from an inbound pass",
                 "defensive_rebound": "after a defensive rebound", "offensive_rebound": "after an offensive rebound",
                 "steal": "after a steal", "turnover": "after a turnover", "made_shot": "after the other team scored",
                 "free_throw": "after a free throw", "period_start": "at the start of the period",
                 "other": "after a change of possession"}
STOPPAGE_PHRASES = {"timeout": "a timeout", "free_throw": "free throws", "period_break": "a break between periods",
                    "out_of_bounds": "the ball went out of bounds", "foul": "a foul was called",
                    "made_basket": "a made basket", "substitution": "a substitution", "unknown": "a stoppage"}


def latest_events(events: Iterable[Event]) -> list[Event]:
    """Latest revision per event_id, retractions removed, sorted by time (types.py revision rule)."""
    best: dict[str, Event] = {}
    for e in events:
        cur = best.get(e.event_id)
        if cur is None or e.revision >= cur.revision:
            best[e.event_id] = e
    return sorted((e for e in best.values() if e.status is not EventStatus.RETRACTED), key=lambda e: (e.t, e.frame_idx))


def _fouled_on_shot(foul: Event, shooter: str | None, t_release: float, t_made: float, w: float) -> bool:
    """The shooter was fouled on this shot: contact from just before the release (gather) to just
    after the ball goes in. Measured against the release too, because a jumper's flight alone can
    exceed ``w``."""
    if foul.kind is not EventKind.FOUL or (shooter is not None and foul.other_player_id not in (shooter, None)):
        return False
    return min(t_release, t_made) - w <= foul.t <= max(t_release, t_made) + w


def flags_from_events(events: Iterable[Event], extra: Iterable[str] = (), cfg: StateConfig = _DEFAULT_CFG) -> list[str]:
    """Highlight/flag codes derivable from event kinds (dunk, block, and-one, steal) plus ``extra``."""
    evs = latest_events(events)
    attempts = {e.shot_id: e for e in evs if e.kind is EventKind.SHOT_ATTEMPT and e.shot_id}
    flags: list[str] = []
    for e in evs:
        if e.kind is EventKind.SHOT_MADE:
            att = attempts.get(e.shot_id or "")
            if att is not None and att.attrs.get("shot_type") == "dunk":
                flags.append("dunk")
            t_release = att.t if att is not None else e.t
            fouled = any(_fouled_on_shot(f, e.player_id, t_release, e.t, cfg.and_one_window_s) for f in evs)
            if fouled and (att is None or att.attrs.get("shot_type") != "free_throw"):
                flags.append("and_one")
        elif e.kind is EventKind.BLOCK:
            flags.append("block")
        elif e.kind is EventKind.STEAL:
            flags.append("steal")
    flags.extend(extra)
    return list(dict.fromkeys(f for f in flags if f in FLAG_PHRASES))


# --------------------------------------------------------------------------- possession summary


@dataclass
class PossessionSummary:
    """One finished possession in local numbers and closed-vocabulary codes (never sent as-is)."""

    offense: str = "the offense"  # team label (team names are not personal data)
    start_cause: str = "other"  # POSSESSION attrs["cause"] vocabulary
    end: str = "other"  # "made_shot" | "missed_shot" | "turnover" | "free_throws" | "other"
    shot_value: int | None = None  # 1/2/3 of the terminal attempt
    shot_type: str | None = None  # SHOT_ATTEMPT attrs["shot_type"]
    shot_distance_ft: float | None = None
    catch_and_shoot: bool | None = None
    terminal_dribbles: int | None = None  # dribbles by the terminal player since their catch
    passes: int = 0  # completed passes since the possession (or last offensive rebound) began
    seconds_since_start: float | None = None  # possession start -> terminal action
    seconds_since_oreb: float | None = None  # last offensive rebound -> terminal action; None if none
    turnover_type: str | None = None
    shot_clock_at_end_s: float | None = None
    game_clock_at_end_s: float | None = None
    defense_set: bool | None = None  # when the attack began, if known from tracking
    flags: list[str] = field(default_factory=list)  # FLAG_PHRASES codes
    actions: list[str] = field(default_factory=list)  # role-only phrases, oldest first, terminal last


def _distance_phrase(dist: float | None, value: int | None, cfg: StateConfig) -> str:
    if value == 3:
        return "deep three-point" if dist is not None and dist >= cfg.deep_three_ft else "three-point"
    if dist is None:
        return "two-point"
    if dist <= cfg.rim_distance_ft:
        return "close-range"
    return "short-range" if dist <= cfg.short_range_ft else "mid-range"


def _dribble_phrase(n: int) -> str:
    return "once" if n == 1 else ("a few times" if n <= 3 else "many times")


def _shot_phrase(p: PossessionSummary, cfg: StateConfig) -> str:
    kind = SHOT_TYPE_PHRASES.get(p.shot_type or "jumper", "shot")
    cns = "catch-and-shoot " if p.catch_and_shoot else ""
    return f"{cns}{_distance_phrase(p.shot_distance_ft, p.shot_value, cfg)} {kind}"


def end_text(p: PossessionSummary, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Closed-vocabulary phrase for how the possession ended (role-only)."""
    if p.end == "made_shot":
        return f"the shooter made a {_shot_phrase(p, cfg)}"
    if p.end == "missed_shot":
        return f"the shooter missed a {_shot_phrase(p, cfg)}"
    if p.end == "turnover":
        return f"the ball handler lost the ball: {TURNOVER_PHRASES.get(p.turnover_type or 'other', 'a turnover')}"
    if p.end == "free_throws":
        return "the offense went to the free-throw line"
    return "the possession ended another way"


def event_text(p: PossessionSummary) -> str:
    """``last_play.event`` closed vocabulary."""
    value = {1: "free throw", 2: "two-point shot", 3: "three-point shot"}.get(p.shot_value or 2, "shot")
    if p.end == "made_shot":
        return f"made {value}"
    if p.end == "missed_shot":
        return f"missed {value}"
    if p.end == "turnover":
        return "turnover"
    if p.end == "free_throws":
        return "free throws"
    return "other"


def terminal_role(p: PossessionSummary, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Closed-vocabulary role of the player who ended the possession."""
    if (p.seconds_since_oreb is not None and p.end in ("made_shot", "missed_shot")
            and p.seconds_since_oreb <= cfg.putback_phrase_s):
        return "shot right after an offensive rebound"
    if p.catch_and_shoot:
        return "caught the ball and shot without dribbling"
    if p.end == "turnover":
        return "the ball handler when the ball was lost"
    if (p.terminal_dribbles or 0) >= 2:
        return "created the shot off the dribble"
    return "ended the possession"


def _segment_actions(evs: list[Event], team: str, cause: str, terminal: Event | None) -> list[str]:
    """Role-only action phrases in time order: passes, dribble runs, blocked shots, offensive rebounds."""
    actions = [f"{team} gained the ball {START_PHRASES.get(cause, START_PHRASES['other'])}"]
    holder, count = None, 0
    for e in evs:
        if e.kind is EventKind.DRIBBLE:
            if e.player_id != holder and count:
                actions.append(f"the ball handler dribbled {_dribble_phrase(count)}")
                count = 0
            holder, count = e.player_id, count + 1
            continue
        if count and e.kind in (EventKind.PASS, EventKind.REBOUND, EventKind.SHOT_ATTEMPT, EventKind.TURNOVER,
                                EventKind.BLOCK):
            actions.append(f"the ball handler dribbled {_dribble_phrase(count)}")
            holder, count = None, 0
        if e.kind is EventKind.PASS and e.attrs.get("completed", True):
            last = terminal is not None and e.other_player_id == terminal.player_id
            actions.append("a pass to the player who ended the possession" if last else "a pass to a teammate")
        elif e.kind is EventKind.BLOCK:
            actions.append("a defender blocked a shot")
        elif e.kind is EventKind.REBOUND and e.attrs.get("offensive"):
            actions.append("a teammate grabbed an offensive rebound")
    if count:
        actions.append(f"the ball handler dribbled {_dribble_phrase(count)}")
    return actions


def _is_free_throw(e: Event) -> bool:
    return e.kind is EventKind.SHOT_ATTEMPT and (e.attrs.get("shot_type") == "free_throw" or e.value == 1)


def _is_play_end(e: Event, offense: Team) -> bool:
    """A field-goal attempt or turnover by the offense."""
    if e.kind is EventKind.SHOT_ATTEMPT:
        return e.team is offense and not _is_free_throw(e)
    return e.kind is EventKind.TURNOVER and e.team in (offense, None)


def _terminal_event(evs: list[Event], offense: Team) -> Event | None:
    """Synergy credits the action that ends the possession: a field-goal attempt, a turnover, or a
    foul drawn by the offense (basketball_rules_stats.md §5). A fouled shooter who misses has no FGA,
    so the foul ends it. A made field goal always ends the possession, so a later defensive foul
    (the and-one, however long the ball was in the air) leaves the made attempt as the end."""
    play = next((e for e in reversed(evs) if _is_play_end(e, offense)), None)
    foul = next((e for e in reversed(evs) if e.kind is EventKind.FOUL and e.team is offense.other()), None)
    if foul is None or (play is not None and foul.t <= play.t):
        return play
    if play is not None and play.kind is EventKind.SHOT_ATTEMPT and play.shot_id:
        if any(e.kind is EventKind.SHOT_MADE and e.shot_id == play.shot_id for e in evs):
            return play
    return foul


def summarize_possession(events: Iterable[Event], offense: Team, team_label: str, *,
                         shot_clock_at_end_s: float | None = None, game_clock_at_end_s: float | None = None,
                         extra_flags: Iterable[str] = (), cfg: StateConfig | None = None) -> PossessionSummary:
    """Summarise one possession's events (shared event semantics) into a ``PossessionSummary``.

    ``events`` should cover the possession (from its POSSESSION event to its end, results included);
    revisions and retractions are resolved first. Events after the opponent's next POSSESSION
    belong to the next possession and are ignored. An offensive rebound keeps the possession and
    restarts the pass/dribble segment. The terminal action is the last field-goal attempt or
    turnover by ``offense``, or the last foul drawn by it (then free throws, if any follow).
    """
    cfg = cfg or _DEFAULT_CFG
    evs = latest_events(events)
    starts = [e for e in evs if e.kind is EventKind.POSSESSION and e.team is offense]
    start = starts[-1] if starts else None
    t_start = start.t if start else (evs[0].t if evs else 0.0)
    cause = str(start.attrs.get("cause", "other")) if start else "other"
    t_stop = min((e.t for e in evs if e.kind is EventKind.POSSESSION and e.team is offense.other()
                  and e.t > t_start), default=math.inf)
    seg_all = [e for e in evs if t_start <= e.t <= t_stop]
    terminal = _terminal_event(seg_all, offense)
    t_end = terminal.t if terminal else (seg_all[-1].t if seg_all else t_start)
    mine = [e for e in seg_all if e.team in (offense, None) and e.t <= t_end]
    orebs = [e.t for e in mine if e.kind is EventKind.REBOUND and e.team is offense and e.attrs.get("offensive")]
    seg_start = max([t_start, *orebs])
    p = PossessionSummary(offense=team_label, start_cause=cause, shot_clock_at_end_s=shot_clock_at_end_s,
                          game_clock_at_end_s=game_clock_at_end_s)
    p.seconds_since_start = t_end - t_start if terminal else None
    p.seconds_since_oreb = (t_end - orebs[-1]) if (orebs and terminal) else None
    window = [e for e in mine if seg_start <= e.t]
    p.passes = sum(1 for e in window if e.kind is EventKind.PASS and e.attrs.get("completed", True))
    if terminal is None:
        p.end = "other"
    elif terminal.kind is EventKind.SHOT_ATTEMPT:
        _fill_shot(p, terminal, window, evs)
    elif terminal.kind is EventKind.TURNOVER:
        p.end, p.turnover_type = "turnover", str(terminal.attrs.get("type", "other"))
    else:  # foul drawn by the offense
        fts = any(_is_free_throw(e) and e.team is offense and e.t >= terminal.t for e in seg_all)
        p.end = "free_throws" if fts else "other"
    p.flags = flags_from_events(seg_all, extra_flags, cfg)
    blocks = [e for e in seg_all if e.kind is EventKind.BLOCK]
    terminal_blocked = (terminal is not None and terminal.kind is EventKind.SHOT_ATTEMPT
                        and any(b.t >= terminal.t and (not b.shot_id or not terminal.shot_id
                                                       or b.shot_id == terminal.shot_id) for b in blocks))
    earlier = sorted({e.event_id: e for e in (*mine, *(b for b in blocks if b.t < t_end))}.values(),
                     key=lambda e: (e.t, e.frame_idx))
    actions = _segment_actions(earlier, team_label, cause, terminal)
    # The terminal action stays the LAST item: fit_state drops the oldest actions first.
    final = terminal_action_text(p, cfg)
    actions.append(f"{final}, and a defender blocked it" if terminal_blocked else final)
    p.actions = actions
    return p


def terminal_action_text(p: PossessionSummary, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """The action that ended the possession (its result lives in ``possession.end``)."""
    if p.end in ("made_shot", "missed_shot"):
        return f"the player who ended the possession took a {_shot_phrase(p, cfg)}"
    if p.end == "turnover":
        return "the ball handler lost the ball"
    if p.end == "free_throws":
        return "the offense drew a foul"
    return "the possession ended without a shot"


def _fill_shot(p: PossessionSummary, att: Event, window: list[Event], evs: list[Event]) -> None:
    # attrs come from CV code and may hold numpy scalars or NaN: normalise before any rule reads them.
    p.shot_value, p.shot_type = int(att.value) or None, att.attrs.get("shot_type")
    p.shot_distance_ft = finite(att.attrs.get("distance_ft"))
    cns = att.attrs.get("catch_and_shoot")
    p.catch_and_shoot = None if cns is None else bool(cns)
    dribbles = finite(att.attrs.get("dribbles_before"))
    if dribbles is not None:
        p.terminal_dribbles = max(0, int(dribbles))
    else:
        catches = [e.t for e in window if e.kind is EventKind.PASS and e.other_player_id == att.player_id
                   and e.attrs.get("completed", True)]
        since = catches[-1] if catches else (window[0].t if window else att.t)
        p.terminal_dribbles = sum(1 for e in window if e.kind is EventKind.DRIBBLE
                                  and e.player_id == att.player_id and since <= e.t <= att.t)
    made = bool(att.shot_id) and any(e.kind is EventKind.SHOT_MADE and e.shot_id == att.shot_id for e in evs)
    p.end = "made_shot" if made else "missed_shot"


def describe_play(p: PossessionSummary, cfg: StateConfig = _DEFAULT_CFG) -> str:
    """Role-only <= 20-word ``last_play.description`` (D16: never names)."""
    if p.end in ("made_shot", "missed_shot"):
        verb = "made" if p.end == "made_shot" else "missed"
        return f"{p.offense}'s shooter {verb} a {_shot_phrase(p, cfg)}"
    if p.end == "turnover":
        return f"{p.offense}'s ball handler lost the ball: {TURNOVER_PHRASES.get(p.turnover_type or 'other', 'a turnover')}"
    if p.end == "free_throws":
        return f"{p.offense} drew a foul and went to the free-throw line"
    return f"{p.offense}'s possession ended"


# --------------------------------------------------------------------------- game situation


@dataclass
class GameSituation:
    """Game context in local numbers; ``situation_block`` turns it into named buckets."""

    home: str = "Home"
    away: str = "Away"
    home_score: int = 0
    away_score: int = 0
    period: int | None = None
    clock_s: float | None = None  # game clock remaining in the period
    league: str = "nba"
    n_periods: int = 4  # regulation periods (RuleSpec.n_periods)
    halves: bool = False
    shot_clock_full_s: int | None = 24  # RuleSpec.shot_clock_s; None = no shot clock
    run_team: str | None = None
    run_points: int = 0
    run_opp_points: int = 0
    lead_change: bool = False
    milestone: bool = False  # a code-flagged milestone is within reach
    foul_note: str | None = None  # pre-rendered and pseudonymised (tokens or roles)

    @property
    def last_period_or_ot(self) -> bool:
        return self.period is not None and self.period >= self.n_periods

    @property
    def diff(self) -> int:
        return int(self.home_score) - int(self.away_score)

    @property
    def leader(self) -> str | None:
        return None if self.diff == 0 else (self.home if self.diff > 0 else self.away)


def situation_block(sit: GameSituation | Mapping[str, Any], cfg: StateConfig = _DEFAULT_CFG) -> dict[str, Any]:
    """§6.1 ``situation``. A Mapping is taken as an already-built block (copied, ``_keys`` dropped)."""
    if isinstance(sit, Mapping):
        return {k: copy.deepcopy(v) for k, v in sit.items() if not str(k).startswith("_")}
    block: dict[str, Any] = {
        "league": LEAGUE_TEXT.get(sit.league, sit.league),
        "period": period_text(sit.period, sit.n_periods, sit.halves),
        "time_bucket": time_bucket(sit.last_period_or_ot, sit.clock_s, sit.shot_clock_full_s, cfg),
        "margin": margin_text(sit.leader, sit.diff, cfg),
        "clutch": clutch_text(sit.last_period_or_ot, sit.clock_s, sit.diff, cfg),
        "run": run_text(sit.run_team, sit.run_points, sit.run_opp_points, cfg),
        "lead_change": LEAD_CHANGE_TEXT if sit.lead_change else None,
        "foul_note": sit.foul_note,
    }
    return {k: v for k, v in block.items() if v is not None}


def situation_local(sit: GameSituation, cfg: StateConfig = _DEFAULT_CFG) -> dict[str, Any]:
    """Bucket codes for the deterministic stakes fallback."""
    return {
        "clutch": is_clutch(sit.last_period_or_ot, sit.clock_s, sit.diff, cfg),
        "margin_code": margin_code(sit.diff, cfg),
        "time_code": time_code(sit.last_period_or_ot, sit.clock_s, sit.shot_clock_full_s, cfg),
        "run_points": int(sit.run_points) if is_run(sit.run_points, sit.run_opp_points, cfg) else 0,
        "lead_change": bool(sit.lead_change),
        "milestone": bool(sit.milestone),
    }


# --------------------------------------------------------------------------- redaction and size


def _name_patterns(names: Iterable[str]) -> list[re.Pattern[str]]:
    full, parts = set(), set()
    for n in names:
        n = (n or "").strip()
        if len(n) < 2:
            continue
        full.add(n)
        parts.update(w for w in re.split(r"\s+", n) if len(w) >= 3)
    ordered = sorted(full, key=len, reverse=True) + sorted(parts - full, key=len, reverse=True)
    return [re.compile(rf"(?<!\w){re.escape(n)}(?!\w)", re.IGNORECASE) for n in ordered]


def redact(text: str, names: Iterable[str], replacement: str = "a player") -> str:
    """Replace every listed name (and each name part of 3+ letters) with a neutral role."""
    for pat in _name_patterns(names):
        text = pat.sub(replacement, text)
    return text


def redact_state(obj: Any, names: Iterable[str], replacement: str = "a player") -> Any:
    """``redact`` applied to every string value in a nested state (keys are left alone)."""
    pats = _name_patterns(names)
    if not pats:
        return obj

    def walk(o: Any) -> Any:
        if isinstance(o, str):
            for pat in pats:
                o = pat.sub(replacement, o)
            return o
        if isinstance(o, Mapping):
            return {k: (v if str(k).startswith("_") else walk(v)) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return walk(obj)


def split_local(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(wire part, ``_local`` part) of a state dict."""
    wire = {k: v for k, v in state.items() if not str(k).startswith("_")}
    local = state.get(LOCAL_KEY)
    return wire, dict(local) if isinstance(local, Mapping) else {}


def _get_path(obj: Any, path: Sequence[str]) -> Any:
    for p in path:
        if not isinstance(obj, Mapping) or p not in obj:
            return None
        obj = obj[p]
    return obj


def _truncate_longest(obj: Any, protect: Sequence[Sequence[str]] = (), min_len: int = 24) -> bool:
    """Shorten the longest string outside ``protect`` subtrees and ``_keys``; False if none can shrink."""
    best: tuple[int, Any, Any] | None = None
    protected = {tuple(p) for p in protect}

    def scan(o: Any, path: tuple) -> None:
        nonlocal best
        items = o.items() if isinstance(o, dict) else enumerate(o) if isinstance(o, list) else ()
        for k, v in items:
            sub = (*path, k)
            if (isinstance(k, str) and k.startswith("_")) or sub in protected:
                continue
            if isinstance(v, str) and len(v) > min_len and (best is None or len(v) > best[0]):
                best = (len(v), o, k)
            elif isinstance(v, (dict, list)):
                scan(v, sub)

    scan(obj, ())
    if best is None:
        return False
    n, container, key = best
    text = container[key]
    target = max(min_len, int(n * 0.8))
    cut = text[:target].rsplit(" ", 1)[0].rstrip(" ,;:")
    # Every step must shorten the string, or fit_state would loop forever on text with no space
    # in the cut window (CJK text, long hyphenated labels, tokens): fall back to a hard cut.
    if len(cut) + 1 >= n:
        cut = text[:target]
    if len(cut) + 1 >= n:
        cut = text[: n - 2]
    container[key] = cut + "…"
    return True


def fit_state(state: dict[str, Any], budget: int, droppable: Sequence[Sequence[str]] = (),
              protect: Sequence[Sequence[str]] = ()) -> dict[str, Any]:
    """Trim ``state`` in place to ``budget`` wire bytes: first drop the oldest items of the
    ``droppable`` lists (each keeps its last item), then shorten the longest strings outside
    the ``protect`` subtrees (e.g. the B4 sentences under review are never altered)."""
    for path in droppable:
        lst = _get_path(state, path)
        while isinstance(lst, list) and len(lst) > 1 and state_bytes(state) > budget:
            lst.pop(0)
    while state_bytes(state) > budget and _truncate_longest(state, protect):
        pass
    if state_bytes(state) > budget:
        log.warning("jev state still %d bytes over a %d-byte budget", state_bytes(state) - budget, budget)
    return state


def _finish(state: dict[str, Any], bundle: str, names: Iterable[str], cfg: StateConfig,
            droppable: Sequence[Sequence[str]] = (), protect: Sequence[Sequence[str]] = ()) -> dict[str, Any]:
    names = list(names)
    if names:
        state = redact_state(state, names)
        state.setdefault(LOCAL_KEY, {})["redact_names"] = names
    return fit_state(state, cfg.max_state_bytes.get(bundle, 1024), droppable, protect)


# --------------------------------------------------------------------------- B1-B5 builders


def last_play_block(p: PossessionSummary, description: str | None = None) -> dict[str, Any]:
    """§6.1 ``last_play`` (role-only description, closed-vocabulary event and flags)."""
    block: dict[str, Any] = {
        "description": description or describe_play(p),
        "event": event_text(p),
    }
    if p.flags:
        block["flags"] = [FLAG_PHRASES[f] for f in p.flags if f in FLAG_PHRASES]
    return block


def possession_block(p: PossessionSummary, cfg: StateConfig = _DEFAULT_CFG) -> dict[str, Any]:
    """§6.1 ``possession``: start, defence-set flag, shot-clock bucket, <= 6 actions, end, role."""
    block: dict[str, Any] = {"start": START_PHRASES.get(p.start_cause, START_PHRASES["other"])}
    if p.defense_set is not None:
        block["defense_set_when_attack_began"] = "yes" if p.defense_set else "no"
    sc = shot_clock_text(p.shot_clock_at_end_s, cfg)
    if sc:
        block["shot_clock_at_end"] = sc
    block["actions"] = list(p.actions[-cfg.max_actions:]) if p.actions else [end_text(p, cfg)]
    block["end"] = end_text(p, cfg)
    block["terminal_player_role"] = terminal_role(p, cfg)
    return block


def possession_local(p: PossessionSummary) -> dict[str, Any]:
    """Numeric play-type features for the deterministic fallback (stays on the device)."""
    d = asdict(p)
    for k in ("actions", "offense"):
        d.pop(k)
    return d


def b1_possession_end(situation: GameSituation | Mapping[str, Any], poss: PossessionSummary, *,
                      description: str | None = None, names: Iterable[str] = (),
                      cfg: StateConfig | None = None) -> dict[str, Any]:
    """B1 ``possession_end`` state (§6.1) with ``_local`` play-type and stakes features."""
    cfg = cfg or _DEFAULT_CFG
    local = possession_local(poss)
    if isinstance(situation, GameSituation):
        local.update(situation_local(situation, cfg))
    state = {"situation": situation_block(situation, cfg), "last_play": last_play_block(poss, description),
             "possession": possession_block(poss, cfg), LOCAL_KEY: local}
    return _finish(state, "B1", names, cfg, droppable=[("possession", "actions")])


def stoppage_text(kind: str, team: str | None = None) -> str:
    """B2 ``stoppage`` phrase from a DeadBallDetector label ("timeout", "free_throw", ...)."""
    if kind == "timeout" and team:
        return f"timeout called by {team}"
    return STOPPAGE_PHRASES.get(kind, STOPPAGE_PHRASES["unknown"])


def b2_dead_ball_fan(situation: GameSituation | Mapping[str, Any], *, stoppage: str,
                     last_play: Mapping[str, Any] | None = None, material: Mapping[str, str] | None = None,
                     storylines: Sequence[str] = (), recent_topics: Sequence[str] = (),
                     storylines_due_now: Sequence[bool] = (), dead_ball_id: str | None = None,
                     names: Iterable[str] = (), cfg: StateConfig | None = None) -> dict[str, Any]:
    """B2 ``dead_ball_fan`` state (§6.2). ``material`` holds code-rendered one-liners per kind."""
    cfg = cfg or _DEFAULT_CFG
    state: dict[str, Any] = {"situation": situation_block(situation, cfg)}
    if last_play:
        state["last_play"] = dict(last_play)
    state["stoppage"] = stoppage
    mat = {k: v for k, v in (material or {}).items() if k in ("stat_note", "tendency") and v}
    if mat:
        state["material"] = mat
    stories = list(storylines)[: cfg.max_storylines]
    if stories:
        state["storylines"] = stories
    if recent_topics:
        state["recent_topics"] = list(recent_topics)[-cfg.max_recent:]
    state[LOCAL_KEY] = {"storylines_due_now": [bool(x) for x in list(storylines_due_now)[: len(stories)]],
                        "dead_ball_id": dead_ball_id}
    return _finish(state, "B2", names, cfg, droppable=[("recent_topics",)])


def b3_dead_ball_coach(situation: GameSituation | Mapping[str, Any], cards: Sequence[str], *, seed: Any,
                       staff_team: str | None = None, timeouts: str | None = None, bonus: str | None = None,
                       names: Iterable[str] = (), cfg: StateConfig | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    """B3 ``dead_ball_coach`` state (§6.3) and ``{card_key: candidate index}``.

    Cards are shuffled with ``random.Random(str(seed))`` (seed = dead-ball id, reproducible), then
    given neutral letter keys in shuffled order, so card order is unbiased and keys carry no content.
    """
    cfg = cfg or _DEFAULT_CFG
    cards = list(cards)[: cfg.max_cards]
    order = list(range(len(cards)))
    random.Random(str(seed)).shuffle(order)
    key_map = {CARD_KEYS[pos]: idx for pos, idx in enumerate(order)}
    sit = situation_block(situation, cfg)
    for k, v in (("staff_team", staff_team), ("timeouts", timeouts), ("bonus", bonus)):
        if v:
            sit[k] = v
    texts = {key: _clip(cards[idx], cfg.max_card_chars) for key, idx in key_map.items()}
    state = {"situation": sit, "cards": texts, LOCAL_KEY: {"dead_ball_id": seed}}
    return _finish(state, "B3", names, cfg), key_map


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


# FactPacket keys a B4 guard may see (a projection: internal keys like allowed_numbers stay local).
GUARD_FACT_KEYS = ("game_clock", "clock", "score", "event", "recent_events", "players_on_court", "players",
                   "teams", "run", "derived", "storylines", "labels")


def _textify(obj: Any) -> Any:
    """Numbers -> text (numpy scalars too); enum keys/values -> their values; ``_keys`` dropped."""
    if obj is None or isinstance(obj, (bool, np.bool_)):
        return None if obj is None else bool(obj)
    if isinstance(obj, enum.Enum):
        return _textify(obj.value)
    if isinstance(obj, numbers.Integral):
        return str(int(obj))
    if isinstance(obj, numbers.Real):
        f = finite(obj)
        return None if f is None else f"{f:g}"
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            key = str(k.value) if isinstance(k, enum.Enum) else str(k)
            if not key.startswith("_"):
                out[key] = _textify(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_textify(v) for v in obj]
    return obj


def guard_facts(facts: Mapping[str, Any], cfg: StateConfig = _DEFAULT_CFG) -> dict[str, Any]:
    """Project a FactPacket onto the B4 ``facts`` block (§6.5); numbers become text Jev only reads."""
    def present(v: Any) -> bool:  # identity/len checks: `v in (None, [], {})` breaks on numpy arrays
        return v is not None and not (isinstance(v, (str, list, tuple, Mapping)) and len(v) == 0)

    out = {k: _textify(facts[k]) for k in GUARD_FACT_KEYS if k in facts and present(facts[k])}
    for k in ("recent_events", "storylines"):
        if isinstance(out.get(k), list):
            out[k] = out[k][-cfg.max_recent:]
    return out


def b4_line_guard(facts: Mapping[str, Any], sentences: Sequence[str], *, names: Iterable[str] = (),
                  cfg: StateConfig | None = None) -> dict[str, Any]:
    """B4 ``line_guard`` state (§6.5): ``facts`` projection plus ``lines.s1``..``lines.s2``."""
    cfg = cfg or _DEFAULT_CFG
    if not 1 <= len(sentences) <= MAX_GUARD_SENTENCES:
        raise ValueError(f"B4 takes 1..{MAX_GUARD_SENTENCES} sentences, got {len(sentences)}")
    f = guard_facts(facts, cfg)
    state: dict[str, Any] = {"facts": f, "lines": {f"s{i}": s for i, s in enumerate(sentences, 1)}}
    budget = cfg.max_state_bytes.get("B4", 1024)
    if state_bytes(state) > budget and isinstance(f.get("players"), Mapping):
        f["players"] = sorted(f["players"])  # keep who is in the facts, drop their counting stats
    return _finish(state, "B4", names, cfg,
                   droppable=[("facts", "recent_events"), ("facts", "derived"), ("facts", "storylines"),
                              ("facts", "players")], protect=[("lines",)])


def drop_speed_text(frac: float | None) -> str:
    """Ball-after-rim cue from the drop-speed fraction of free fall (10-90% => net contact;
    Roboflow shot-evaluator rule via cv_pipeline.md, sourced; cut-offs unverified on our data)."""
    if frac is None:
        return "not observed"
    if frac < 0.1:
        return "barely moved down after reaching the rim"
    if frac <= 0.9:
        return "reappeared below the net, falling slower than free fall"
    return "fell at free-fall speed, as if it touched nothing"


def net_motion_text(z: float | None) -> str:
    """Net-motion cue from a motion z-score (cut-offs unverified; tune on labelled shots)."""
    if z is None:
        return "net not visible"
    if z >= 3.0:
        return "strong net movement right after the ball reached the rim"
    return "slight net movement" if z >= 1.5 else "no net movement"


def detections_text(n_frames: int | None) -> str:
    """``ball-in-basket`` detector cue from the number of frames with a detection."""
    if not n_frames:
        return "not detected"
    return "detected in one frame" if n_frames == 1 else "detected in several frames"


_CUE_BUCKETERS = {"drop_speed_frac": ("ball_after_rim", drop_speed_text),
                  "net_motion_z": ("net_motion", net_motion_text),
                  "ball_in_basket_frames": ("ball_in_basket_detections", detections_text)}


def b5_event_ambiguity(kind: str, cues: Mapping[str, Any], *, cfg: StateConfig | None = None) -> dict[str, Any]:
    """B5 state (§6.4): ``{"shot": {...}}`` or ``{"release": {...}}`` of closed-vocabulary phrases.

    String cues pass through; booleans become yes/no; None becomes "not observed"; the known numeric
    cues (``drop_speed_frac``, ``net_motion_z``, ``ball_in_basket_frames``) are bucketed; any other
    number is dropped, because Jev must never see raw measurements.
    """
    cfg = cfg or _DEFAULT_CFG
    root = {"shot_outcome": "shot", "release_intent": "release"}.get(kind)
    if root is None:
        raise ValueError(f"unknown B5 kind {kind!r}")
    block: dict[str, str] = {}
    for name, value in cues.items():
        if str(name).startswith("_"):
            continue
        value = sanitize(value)  # numpy bools/floats from the cue model -> Python values, NaN -> None
        if name in _CUE_BUCKETERS:
            field_name, fn = _CUE_BUCKETERS[name]
            block[field_name] = fn(value)
        elif isinstance(value, str):
            block[str(name)] = value
        elif isinstance(value, bool):
            block[str(name)] = "yes" if value else "no"
        elif value is None:
            block[str(name)] = "not observed"
        else:
            log.warning("B5 cue %r dropped: raw numbers are never sent to Jev", name)
    state: dict[str, Any] = {root: block}
    local = cues.get(LOCAL_KEY)
    if isinstance(local, Mapping):
        state[LOCAL_KEY] = dict(local)
    return fit_state(state, cfg.max_state_bytes.get("B5", 614))
