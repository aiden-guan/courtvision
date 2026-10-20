"""Rule-based coach cards (``Insight``) computed from ``GameStats`` and the event stream.

Card contract (narration.md §7.2-§7.3, jev_integration_design.md site 6, RESEARCH.md §2.7):

* Text is <= 12 words, flat and numeric, and carries its sample size. Entities are placeholders,
  ``{player}`` and ``{team}``, rendered locally (``InsightEngine.render``) or pseudonymised before
  any cloud call (RESEARCH.md §7A.7, §7A.9). Cards are information, never instructions: a scoring
  run is never "call timeout" (timeout effects are weak or disputed [50]-[52]).
* ``evidence`` holds the computed numbers, always with ``n`` (sample size) and ``n_unit``.
* Each card has a dedupe key that includes its value (the same fact is never re-sent; a card
  whose condition clears and returns is shown again, not re-sent) and a per-kind cooldown in
  game-clock seconds (one card of a type per cooldown, whatever its subject; a new period clears
  it, and video time is used when the game clock is unknown). ``active()`` returns at most 3
  cards by priority.
* Expected-eFG cards only for leagues in ``xefg_leagues`` (NBA) until per-league priors exist;
  otherwise raw eFG with n (RESEARCH.md §7A.2).

Rules: foul trouble (Q+1, adapted to ``RuleSpec.foul_out`` and halves), hot hand, scoring run,
shot-quality gap, shot diet, offensive-rebounding battle, turnover burst, lineup +/- (>= 20
possessions), fatigue (top-speed decline; needs ``observe_frame``), decision quality, FT rate, and
team fouls / bonus (needs a ``RuleSpec`` and FOUL events).

``stats`` is read by attribute name per the ARCHITECTURE ``GameStats`` contract (a plain dict from
``GameStats.to_dict()`` also works), so this module does not import ``stats.engine``.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import numpy as np

from courtvision.geometry.court import CourtSpec
from courtvision.stats.advanced import efg
from courtvision.types import Event, EventKind, EventStatus, FrameState, Insight, Roster, Team

if TYPE_CHECKING:
    from courtvision.geometry.rules import RuleSpec

__all__ = ["InsightConfig", "InsightEngine", "PenaltyState", "penalty_state", "word_count"]

log = logging.getLogger(__name__)

_TEAMS = (Team.HOME, Team.AWAY)
_TRACKED_KINDS = frozenset({EventKind.FOUL, EventKind.TURNOVER, EventKind.POSSESSION})


@dataclass
class InsightConfig:
    """Thresholds for every card rule. Status per value: sourced / unverified / tuned / proposal."""

    max_active: int = 3  # at most 3 cards visible (narration.md §7.3); proposal
    max_words: int = 12  # <= 12 words per card (narration.md §7.3; RESEARCH §2.7); proposal
    # Higher = more important. Order from jev_integration_design.md site 6 fallback: foul trouble >
    # run >= 8-0 > bonus > shot-quality gap > tendencies (proposal); the rest tuned.
    priorities: dict[str, int] = field(default_factory=lambda: {
        "foul_trouble": 90,
        "run": 80,
        "team_fouls": 70,
        "turnovers": 60,
        "shot_quality": 50,
        "decision": 45,
        "oreb": 40,
        "hot_hand": 35,
        "lineup": 30,
        "fatigue": 30,
        "ft_rate": 25,
        "shot_diet": 20,
    })
    # Same card type <= 1 per 120 s of game clock (narration.md §7.3/§9.5); proposal. Foul trouble
    # and team fouls post immediately (Q+1 alerts bypass cooldowns, site 6) and are deduped by value.
    # Runs are rate-limited by the "update at +4 only" rule instead (§9.5). Slow-moving team
    # aggregates use 300 s (the 5-min same-fact rule, §9.5); tuned.
    default_cooldown_s: float = 120.0
    # The cooldown is per card TYPE ("same alert type <= 1 per 120 s", narration.md §7.3; "a card
    # type appears at most once per 120 s", jev_integration_design.md site 6). True keys it per
    # (type, subject) instead, which lets e.g. five players' decision cards post within seconds.
    cooldown_per_subject: bool = False
    cooldown_s: dict[str, float] = field(default_factory=lambda: {
        "foul_trouble": 0.0,
        "team_fouls": 0.0,
        "run": 0.0,
        "shot_quality": 300.0,
        "shot_diet": 300.0,
        "oreb": 300.0,
        "ft_rate": 300.0,
        "lineup": 300.0,
        "fatigue": 300.0,
    })
    active_ttl_s: float = 600.0  # video seconds a "news" card stays active; tuned
    state_kinds: list[str] = field(default_factory=lambda: ["foul_trouble", "team_fouls"])  # active while true
    # Rule defaults when no RuleSpec is given (rules note §3; sourced): foul-out, periods, halves.
    # "technicals": technical fouls count toward foul-out, as config/rules/*.yaml
    # foul_out_counts_technicals (NFHS Case 4.14.1 sourced; FIBA unverified; others false).
    league_rule_defaults: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "nba": {"foul_out": 6, "n_periods": 4, "period_s": 720, "halves": False, "technicals": False},
        "fiba": {"foul_out": 5, "n_periods": 4, "period_s": 600, "halves": False, "technicals": True},
        "ncaa_m": {"foul_out": 5, "n_periods": 2, "period_s": 1200, "halves": True, "technicals": False},
        "ncaa_w": {"foul_out": 5, "n_periods": 4, "period_s": 600, "halves": False, "technicals": False},
        "nfhs": {"foul_out": 5, "n_periods": 4, "period_s": 480, "halves": False, "technicals": True},
    })
    # Hot hand: makes in the last N FGAs, the latest one made; tuned (descriptive only)
    hot_hand_window: int = 4
    hot_hand_makes: int = 3
    # Runs: 8-0 or 10-2, update every +4 (narration.md §9.5 run callout); proposal
    run_min_points: int = 8
    run_alt_points: int = 10
    run_alt_max_opp: int = 2
    run_update_step: int = 4
    run_window_rearm_s: float = 300.0  # a window run re-arms after this (video s) when the stats lack
    # "window_s"; matches stats-core run_window_s; tuned
    # Shot quality: |eFG - xeFG| >= 10 pts with >= 15 FGA (site 6 fallback); proposal
    xefg_leagues: list[str] = field(default_factory=lambda: ["nba"])  # RESEARCH §7A.2
    sq_min_shots: int = 15
    sq_min_gap: float = 0.10  # also the raw-eFG gap between teams outside xefg_leagues; tuned
    # Shot diet: midrange share of FGA; NBA long-mid pts/shot .824 vs rim 1.345 (rules note §4c)
    diet_min_shots: int = 15
    diet_midrange_share: float = 0.30  # unverified (NBA long 2s are ~7% of FGA, §4c)
    # Offensive rebounding: OREB / (OREB + opp DREB); NBA ORB% 26.0 in 2025-26 (§4a)
    oreb_min_chances: int = 10  # tuned
    oreb_pct_high: float = 0.40  # ~1.5x the NBA rate; unverified
    # Turnover burst: >= 3 in the team's last 5 possessions (POSSESSION events), else in 180 s video
    tov_burst_count: int = 3  # tuned
    tov_burst_possessions: int = 5  # tuned
    tov_burst_window_s: float = 180.0  # fallback without POSSESSION events; tuned
    # Lineup +/-: hidden below ~20 possessions (narration.md §7.2 [59]); proposal
    lineup_min_possessions: int = 20
    lineup_min_abs_net: float = 5.0  # tuned
    # Fatigue: per-player top speed per time bin (observe_frame); tuned / unverified
    fatigue_bin_s: float = 60.0
    fatigue_min_bins: int = 6
    fatigue_recent_bins: int = 3
    fatigue_drop: float = 0.15  # recent top speed >= 15% below the earlier 75th percentile; unverified
    fatigue_min_baseline_fts: float = 14.0  # ignore players who never ran (~9.5 mph); tuned
    fatigue_quantile: float = 95.0  # robust per-bin "top speed" (percentile of samples); tuned
    fatigue_min_coverage_s: float = 20.0  # a bin counts only with this much tracked time; tuned
    fatigue_max_speed_fts: float = 32.0  # drop implausible samples (perception clamps at 30 ft/s)
    # Decision quality: >= 3 "poor" grades in a player's last 8 graded decisions; tuned
    decision_window: int = 8
    decision_min_poor: int = 3
    # FT rate FTA / FGA; NBA FTr .264 in 2025-26 (§4a)
    ft_rate_min_fga: int = 15
    ft_rate_high: float = 0.40  # ~1.5x the NBA rate; unverified
    # Fouls that are not team fouls (coach / bench technicals per the FIBA manual, rules note §2a;
    # player technicals vary by league; unverified)
    team_foul_excluded_types: list[str] = field(default_factory=lambda: ["technical", "coach_technical",
                                                                          "bench_technical"])


# --------------------------------------------------------------------------- helpers


def word_count(text: str) -> int:
    """Whitespace word count; a placeholder such as ``{player}`` counts as one word."""
    return len(text.split())


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _team(value: Any) -> Team | None:
    if isinstance(value, Team):
        return value
    if isinstance(value, str):
        try:
            return Team(value)
        except ValueError:
            return None
    return None


def _num(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _pct(x: float) -> int:
    return int(round(100.0 * x))


def _article(n: int) -> str:
    """'an' before numbers spoken with a vowel sound (8, 11, 18, 80-89, 800-899...)."""
    return "an" if str(n).startswith("8") or n in (11, 18) else "a"


@dataclass(frozen=True)
class _Clock:
    t: float
    period: int | None
    game_clock: float | None

    def elapsed_since(self, earlier: _Clock) -> float:
        """Game-clock seconds within a period (inf across periods); video seconds without a clock."""
        if None not in (self.period, earlier.period, self.game_clock, earlier.game_clock):
            if self.period != earlier.period:
                return math.inf
            return float(earlier.game_clock) - float(self.game_clock)  # remaining time decreases
        return self.t - earlier.t


@dataclass
class _Card:
    kind: str
    subject: str
    key: tuple
    text: str
    evidence: dict[str, Any]
    team: Team | None = None
    player_id: str | None = None
    priority_bonus: int = 0
    transient: bool = False  # expires after active_ttl_s even for a state kind (e.g. fouled out)


@dataclass
class _SpeedBin:
    index: int
    first_t: float
    last_t: float
    speeds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class _Format:
    foul_out: int
    n_periods: int
    period_s: float
    halves: bool
    counts_technicals: bool = False  # technical fouls count toward foul-out (NFHS, FIBA)


# --------------------------------------------------------------------------- team fouls


@dataclass(frozen=True)
class PenaltyState:
    """Team-foul situation of the fouling team in the current bonus scope."""

    fouls: int  # team fouls in the scope
    award: str  # what the NEXT team foul gives the opponent: "none" | "one_and_one" | "two_shots"
    in_bonus: bool  # the opponent already shoots on this team's non-shooting fouls
    two_shots_from: int  # effective double-bonus threshold (late-period quota applied)
    one_and_one_from: int | None
    periods: tuple[int, ...]  # periods whose fouls count together


def _scope_periods(rules: RuleSpec, period: int) -> set[int]:
    n = rules.n_periods
    bonus = rules.bonus or {}
    if period > n:
        if bonus.get("ot_counts_toward_last_period"):
            return _scope_periods(rules, n) | set(range(n + 1, period + 1))
        return {period}
    if bonus.get("scope") == "half" and not rules.halves and n % 2 == 0:
        half = n // 2
        first = 1 if period <= half else half + 1
        return set(range(first, first + half))
    return {period}


def penalty_state(rules: RuleSpec, period: int, game_clock: float | None,
                  fouls: Iterable[tuple[int, float | None]]) -> PenaltyState:
    """Bonus state from one team's fouls ``(period, game_clock_at_foul)`` using ``rules.bonus``.

    Handles period / half scope, OT carry-over (``ot_counts_toward_last_period``) or the OT
    threshold (``ot_two_shots_from``), the one-and-one stage, and the late-period quota
    (``late_period_s`` + ``late_two_shots_from``: when under the limit, the penalty starts at the
    ``late_two_shots_from``-th foul inside the window; needs foul game clocks).
    """
    bonus = rules.bonus or {}
    periods = _scope_periods(rules, period)
    in_scope = [(p, gc) for p, gc in fouls if p in periods]
    count = len(in_scope)
    two = int(bonus["two_shots_from"])
    if period > rules.n_periods and not bonus.get("ot_counts_toward_last_period") and bonus.get("ot_two_shots_from"):
        two = int(bonus["ot_two_shots_from"])
    late_s, late_two = bonus.get("late_period_s"), bonus.get("late_two_shots_from")
    if late_s and late_two and game_clock is not None and game_clock <= float(late_s):
        late = sum(1 for p, gc in in_scope if p == period and gc is not None and gc <= float(late_s))
        two = min(two, (count - late) + int(late_two))
    oao_raw = bonus.get("one_and_one_from")
    oao = int(oao_raw) if oao_raw is not None and int(oao_raw) < two else None
    nxt = count + 1
    if nxt >= two:
        award = "two_shots"
    elif oao is not None and nxt >= oao:
        award = "one_and_one"
    else:
        award = "none"
    in_bonus = count >= (oao if oao is not None else two)
    return PenaltyState(count, award, in_bonus, two, oao, tuple(sorted(periods)))


# --------------------------------------------------------------------------- engine


class InsightEngine:
    """Turns stats and events into deduped, rate-limited coach cards (at most 3 active)."""

    def __init__(self, roster: Roster | None, court: CourtSpec, config: InsightConfig | None = None,
                 rules: RuleSpec | None = None) -> None:
        self.roster = roster or Roster(entries=[])
        self.court = court
        self.config = config or InsightConfig()
        self.rules = rules
        self._counter = 0
        self._events: dict[str, Event] = {}
        self._rev: dict[str, int] = {}
        self._event_ctx: dict[str, tuple[int | None, float | None]] = {}
        self._seen_keys: dict[tuple, Insight] = {}  # dedupe key -> the card emitted for it
        self._last_emit: dict[tuple[str, str], _Clock] = {}  # cooldown slot -> last emission
        self._active: dict[tuple[str, str], Insight] = {}
        self._transient: set[str] = set()  # insight ids that expire after the TTL despite their kind
        self._run_identity: tuple[Team, Any] | None = None
        self._run_episode = 0  # episode counter (new keys per run)
        self._run_key_episode = 0  # episode used in the current run's keys
        self._window_runs: dict[Team, tuple[int, float]] = {}  # team -> (episode, video t it started)
        self._run_announced: dict[Team, tuple[int, float, tuple]] = {}  # team -> (points, video t, key)
        self._open_bins: dict[str, _SpeedBin] = {}
        self._speed_tops: dict[str, list[float]] = {}
        self._rule_fns: list[Callable[[Any, _Clock], list[_Card]]] = [
            self._foul_trouble, self._run, self._team_fouls, self._turnovers, self._shot_quality,
            self._decisions, self._oreb, self._hot_hand, self._lineups, self._fatigue, self._ft_rate,
            self._shot_diet,
        ]

    # ------------------------------------------------------------------ public API

    def update(self, stats: Any, new_events: list[Event], t: float) -> list[Insight]:
        """Ingest events, evaluate every rule, return the newly emitted cards (priority order)."""
        clock = _Clock(float(t), _opt_int(_get(stats, "period")), _opt_float(_get(stats, "game_clock")))
        self._ingest(new_events, clock)
        cards: list[_Card] = []
        for rule in self._rule_fns:
            try:
                cards.extend(rule(stats, clock))
            except Exception:  # noqa: BLE001 - one broken rule must not silence the others
                log.exception("insight rule %s failed", getattr(rule, "__name__", rule))
        live = {(c.kind, c.subject) for c in cards}
        emitted: list[Insight] = []
        for card in sorted(cards, key=lambda c: (-self._priority(c), c.kind, c.subject)):
            ins = self._maybe_emit(card, clock)
            if ins is not None:
                emitted.append(ins)
        self._prune_active(live, clock)
        return emitted

    def active(self) -> list[Insight]:
        """Currently valid cards, highest priority first (at most ``max_active``)."""
        ranked = sorted(self._active.values(), key=lambda i: (-i.priority, -i.t, i.insight_id))
        return ranked[: self.config.max_active]

    def observe_frame(self, frame: FrameState) -> None:
        """Feed player speeds for the fatigue rule (call once per processed frame)."""
        cfg = self.config
        idx = int(math.floor(frame.t / cfg.fatigue_bin_s))
        for p in frame.players:
            speed = p.speed
            if not math.isfinite(speed) or speed > cfg.fatigue_max_speed_fts:
                continue
            cur = self._open_bins.get(p.player_id)
            if cur is not None and cur.index != idx:
                self._close_bin(p.player_id, cur)
                cur = None
            if cur is None:
                cur = self._open_bins[p.player_id] = _SpeedBin(idx, frame.t, frame.t)
            cur.speeds.append(speed)
            cur.last_t = frame.t

    def render(self, insight: Insight) -> str:
        """Local rendering: placeholders -> roster names (never send the result to a cloud API)."""
        player = self.roster.display_name(insight.player_id) if insight.player_id else "player"
        team = self.roster.team_names.get(insight.team, insight.team.value) if insight.team else "team"
        return insight.text.replace("{player}", player).replace("{team}", team)

    # ------------------------------------------------------------------ bookkeeping

    def _ingest(self, events: Iterable[Event], clock: _Clock) -> None:
        for e in events:
            if e.kind not in _TRACKED_KINDS:
                continue
            if e.revision < self._rev.get(e.event_id, -1):
                continue  # stale revision
            self._rev[e.event_id] = e.revision
            if e.status is EventStatus.RETRACTED:
                self._events.pop(e.event_id, None)
                continue
            self._events[e.event_id] = e
            period = _opt_int(e.attrs.get("period"))
            gc = _opt_float(e.attrs.get("game_clock"))
            first = self._event_ctx.get(e.event_id, (clock.period, clock.game_clock))
            self._event_ctx[e.event_id] = (period if period is not None else first[0], gc if gc is not None else first[1])

    def _priority(self, card: _Card) -> int:
        return int(self.config.priorities.get(card.kind, 0)) + card.priority_bonus

    def _maybe_emit(self, card: _Card, clock: _Clock) -> Insight | None:
        slot = (card.kind, card.subject)
        seen = self._seen_keys.get(card.key)
        if seen is not None:
            # Same fact again: never re-sent. If its slot was cleared (e.g. a retracted foul that
            # came back), the card it already produced is shown again, silently.
            if slot not in self._active:
                self._active[slot] = seen
            return None
        cooldown = float(self.config.cooldown_s.get(card.kind, self.config.default_cooldown_s))
        cd_slot = slot if self.config.cooldown_per_subject else (card.kind, "*")
        last = self._last_emit.get(cd_slot)
        if last is not None and cooldown > 0 and clock.elapsed_since(last) < cooldown:
            return None
        if word_count(card.text) > self.config.max_words:
            log.warning("insight %s dropped: %d words > %d", card.kind, word_count(card.text), self.config.max_words)
            return None
        self._counter += 1
        ins = Insight(insight_id=f"i{self._counter:04d}", t=clock.t, kind=card.kind, text=card.text,
                      priority=self._priority(card), team=card.team, player_id=card.player_id,
                      evidence=dict(card.evidence))
        self._seen_keys[card.key] = ins
        self._last_emit[cd_slot] = clock
        self._active[slot] = ins
        if card.transient:
            self._transient.add(ins.insight_id)
        return ins

    def _prune_active(self, live: set[tuple[str, str]], clock: _Clock) -> None:
        for slot, ins in list(self._active.items()):
            sticky = ins.kind in self.config.state_kinds and ins.insight_id not in self._transient
            expired = not sticky and clock.t - ins.t > self.config.active_ttl_s
            if slot not in live or expired:
                del self._active[slot]

    def _format(self) -> _Format:
        if self.rules is not None:
            r = self.rules
            return _Format(int(r.foul_out), int(r.n_periods), float(r.period_s), bool(r.halves),
                           bool(getattr(r, "foul_out_counts_technicals", False)))
        d = self.config.league_rule_defaults.get(self.court.league) or self.config.league_rule_defaults["nba"]
        return _Format(int(d["foul_out"]), int(d["n_periods"]), float(d["period_s"]), bool(d["halves"]),
                       bool(d.get("technicals", False)))

    def _period_label(self, period: int) -> str:
        fmt = self._format()
        if period > fmt.n_periods:
            k = period - fmt.n_periods
            return "OT" if k == 1 else f"OT{k}"
        return f"H{period}" if fmt.halves else f"Q{period}"

    def _quarter_equivalent(self, period: int, game_clock: float | None) -> int:
        """Period expressed in quarters (1-4) so the Q+1 rule works for halves and OT."""
        fmt = self._format()
        if period > fmt.n_periods:
            return 4
        if fmt.halves:
            first_part = game_clock is None or game_clock > fmt.period_s / 2.0
            return 2 * (period - 1) + (1 if first_part else 2)
        return max(1, math.ceil(period * 4 / fmt.n_periods))

    def _team_of_player(self, pid: str | None, fallback: Any = None) -> Team | None:
        """Same order as stats-core's ``TeamResolver``: roster, then the ``home-`` / ``away-`` id
        prefix of provisional ids (types.py), then the event / record team."""
        team = self.roster.team_of(pid)
        if team is None and pid:
            team = _team(str(pid).split("-", 1)[0].lower())
        return team or _team(fallback)

    def _fg_shots(self, stats: Any) -> list[Any]:
        shots = [s for s in (_get(stats, "shots") or []) if _get(s, "made") is not None
                 and int(_num(_get(s, "value"), 0)) in (2, 3)]
        return sorted(shots, key=lambda s: _num(_get(s, "t")))

    def _shots_of(self, shots: list[Any], team: Team) -> list[Any]:
        return [s for s in shots if self._team_of_player(_get(s, "player_id"), _get(s, "team")) is team]

    # ------------------------------------------------------------------ rules

    def _foul_trouble(self, stats: Any, clock: _Clock) -> list[_Card]:
        if clock.period is None:
            return []
        fmt = self._format()
        q = self._quarter_equivalent(clock.period, clock.game_clock)
        threshold = min(q + 1, fmt.foul_out - 1)
        label = self._period_label(clock.period)
        rule = "Q+1" if not fmt.halves else "adapted Q+1"
        cards = []
        for pid, line in (_get(stats, "players") or {}).items():
            tf = int(_num(_get(line, "tf"))) if fmt.counts_technicals else 0
            pf = int(_num(_get(line, "pf"))) + tf  # fouls that count toward foul-out
            if pf < threshold:
                continue
            if pf >= fmt.foul_out:
                text, bonus, key_q = f"{{player}} fouled out with {pf} fouls", -5, "out"
            elif pf == fmt.foul_out - 1:
                text, bonus, key_q = f"{{player}} {pf} fouls in {label}, one from fouling out", 5, q
            else:
                text, bonus, key_q = f"{{player}} {pf} fouls in {label} ({rule})", 0, q
            ev = {"n": pf, "n_unit": "fouls", "pf": pf - tf, "tf": tf, "period": clock.period,
                  "quarter_equivalent": q, "threshold": threshold, "foul_out": fmt.foul_out, "rule": rule}
            # fouled out is news, not a live state: it must not hold one of the 3 slots all game
            cards.append(_Card("foul_trouble", pid, ("foul_trouble", pid, pf, key_q), text, ev,
                               team=self._team_of_player(pid, _get(line, "team")), player_id=pid,
                               priority_bonus=bonus, transient=key_q == "out"))
        return cards

    def _hot_hand(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        by_player: dict[str, list[Any]] = {}
        for s in self._fg_shots(stats):
            pid = _get(s, "player_id")
            if pid is not None:  # a shooter-unknown attempt is nobody's streak (not player "None")
                by_player.setdefault(str(pid), []).append(s)
        cards = []
        for pid, shots in by_player.items():
            last = shots[-cfg.hot_hand_window:]
            made = sum(1 for s in last if _get(s, "made"))
            if len(last) < cfg.hot_hand_window or made < cfg.hot_hand_makes or not _get(last[-1], "made"):
                continue
            xfgs = [_num(_get(s, "xfg")) for s in last if _get(s, "xfg") is not None]
            ev = {"n": len(last), "n_unit": "shots", "made": made, "game_fgm": sum(1 for s in shots if _get(s, "made")),
                  "game_fga": len(shots), "mean_xfg_window": float(np.mean(xfgs)) if xfgs else None}
            text = f"{{player}} made {made} of last {len(last)} shots"
            team = self._team_of_player(pid, _get(last[-1], "team"))
            cards.append(_Card("hot_hand", pid, ("hot_hand", pid, _get(last[-1], "shot_id")), text, ev,
                               team=team, player_id=pid))
        return cards

    def _run(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        found = self._find_run(_get(stats, "run") or {})
        if found is None:
            self._run_identity = None
            return []
        team, pts, opp, since, source = found
        run = _get(stats, "run") or {}
        identity = (team, since if source == "unanswered" else "window")
        if identity != self._run_identity:  # a new run: new keys; one run is rate-limited by +4 steps
            self._run_identity = identity
            self._run_key_episode = self._new_run_episode(team, source, run, clock)
        bucket = (pts - cfg.run_min_points) // max(cfg.run_update_step, 1)
        key = ("run", team.value, self._run_key_episode, bucket)
        last = self._run_announced.get(team)
        span = _opt_float(_get(run, "window_s")) or cfg.run_window_rearm_s
        if key not in self._seen_keys:
            # The same stretch re-framed (the unanswered 8-0 inside an announced 13-2, or a 14-0
            # right after "14-2") is not news unless it is bigger: keep the earlier card up. An
            # unanswered run that started after the last announcement is a new stretch.
            same_stretch = source == "window" or (since is not None and last is not None and since <= last[1])
            if last is not None and same_stretch and pts <= last[0] and clock.t - last[1] < span:
                return [_Card("run", team.value, last[2], "", {}, team=team)] if last[2] in self._seen_keys else []
            self._run_announced[team] = (pts, clock.t, key)
        duration = _opt_float(_get(run, "window_s")) if source == "window" else (
            None if since is None else clock.t - since)
        ev = {"n": pts + opp, "n_unit": "points", "points": pts, "opp_points": opp, "source": source,
              "since_t": since, "duration_s": duration}
        text = f"{{team}} on {_article(pts)} {pts}-{opp} run"
        return [_Card("run", team.value, key, text, ev, team=team)]

    def _new_run_episode(self, team: Team, source: str, run: Any, clock: _Clock) -> int:
        """Episode for a run that just (re)qualified. A window run (last-N-seconds differential)
        that drops below the bar when an old basket leaves the window and re-qualifies on the next
        one is the same stretch of play: it keeps its episode until the window has fully turned
        over, so the same 10-2 is not announced twice."""
        if source == "window":
            prev = self._window_runs.get(team)
            span = _opt_float(_get(run, "window_s")) or self.config.run_window_rearm_s
            if prev is not None and clock.t - prev[1] < span:
                return prev[0]
        self._run_episode += 1
        if source == "window":
            self._window_runs[team] = (self._run_episode, clock.t)
        return self._run_episode

    def _find_run(self, run: Any) -> tuple[Team, int, int, float | None, str] | None:
        """(team, points, opp_points, since_t, source) of a qualifying run, or None.

        Reads the contract's ``{"team", "points", "opp_points", "since_t"}`` (the current run) and,
        when present, stats-core's last-N-seconds ``window`` / ``window_leader`` (for 10-2 style runs).
        """
        cfg = self.config

        def qualifies(pts: int, opp: int) -> bool:
            return (pts >= cfg.run_min_points and opp == 0) or (pts >= cfg.run_alt_points and opp <= cfg.run_alt_max_opp)

        found: list[tuple[Team, int, int, float | None, str]] = []
        team = _team(_get(run, "team"))
        pts, opp = int(_num(_get(run, "points"))), int(_num(_get(run, "opp_points")))
        if team is not None and qualifies(pts, opp):
            found.append((team, pts, opp, _opt_float(_get(run, "since_t")), "unanswered"))
        leader, window = _team(_get(run, "window_leader")), _get(run, "window")
        if leader is not None and isinstance(window, Mapping):
            lead, other = int(_num(_team_keyed(window, leader))), int(_num(_team_keyed(window, leader.other())))
            if qualifies(lead, other):
                found.append((leader, lead, other, None, "window"))
        # The bigger view wins: an unanswered 8-0 inside a 13-2 window is the same stretch, and
        # reporting the 8-0 after the 13-2 would walk the news backwards. Ties keep "unanswered".
        return max(found, key=lambda r: r[1]) if found else None

    def _xefg_allowed(self) -> bool:
        """Expected-eFG only when both the court and the RuleSpec (if given) name a priors league:
        a game set up with an NBA court but an NFHS RuleSpec still gets raw eFG (RESEARCH §7A.2)."""
        leagues = self.config.xefg_leagues
        if self.court.league not in leagues:
            return False
        return self.rules is None or str(getattr(self.rules, "ruleset", "")) in leagues

    def _shot_quality(self, stats: Any, clock: _Clock) -> list[_Card]:
        shots = self._fg_shots(stats)
        if self._xefg_allowed():
            return [c for team in _TEAMS if (c := self._xefg_card(self._shots_of(shots, team), team))]
        return self._raw_efg_cards(shots)

    def _xefg_card(self, shots: list[Any], team: Team) -> _Card | None:
        cfg = self.config
        rated = [s for s in shots if _get(s, "xfg") is not None]
        n = len(rated)
        if n < cfg.sq_min_shots:
            return None
        fgm = sum(1 for s in rated if _get(s, "made"))
        fg3m = sum(1 for s in rated if _get(s, "made") and int(_num(_get(s, "value"))) == 3)
        actual = efg(fgm, fg3m, n) or 0.0
        expected = sum(_num(_get(s, "xfg")) * int(_num(_get(s, "value"))) / 2.0 for s in rated) / n
        gap = actual - expected
        if abs(gap) < cfg.sq_min_gap:
            return None
        ev = {"n": n, "n_unit": "shots", "efg": actual, "xefg": expected, "gap": gap, "fgm": fgm, "fg3m": fg3m,
              "model": "xeFG (NBA priors)"}
        text = f"{{team}} eFG {_pct(actual)}% on expected {_pct(expected)}% (n={n})"
        return _Card("shot_quality", team.value, ("shot_quality", team.value, n // 5, gap > 0), text, ev, team=team)

    def _raw_efg_cards(self, shots: list[Any]) -> list[_Card]:
        cfg = self.config
        stats_by_team = {}
        for team in _TEAMS:
            ts = self._shots_of(shots, team)
            fgm = sum(1 for s in ts if _get(s, "made"))
            fg3m = sum(1 for s in ts if _get(s, "made") and int(_num(_get(s, "value"))) == 3)
            stats_by_team[team] = (len(ts), efg(fgm, fg3m, len(ts)))
        (n_h, e_h), (n_a, e_a) = stats_by_team[Team.HOME], stats_by_team[Team.AWAY]
        if min(n_h, n_a) < cfg.sq_min_shots or e_h is None or e_a is None or abs(e_h - e_a) < cfg.sq_min_gap:
            return []
        team = Team.HOME if e_h > e_a else Team.AWAY
        (n, e), (n_o, e_o) = stats_by_team[team], stats_by_team[team.other()]
        ev = {"n": n, "n_unit": "shots", "efg": e, "opp_efg": e_o, "opp_n": n_o,
              "model": "raw eFG (no league priors; RESEARCH §7A.2)"}
        text = f"{{team}} eFG {_pct(e)}% vs opponent {_pct(e_o)}% (n={n})"
        return [_Card("shot_quality", team.value, ("shot_quality", team.value, n // 5, "raw"), text, ev, team=team)]

    def _shot_diet(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        all_shots = self._fg_shots(stats)
        cards = []
        for team in _TEAMS:
            shots = [s for s in self._shots_of(all_shots, team) if _get(s, "zone")]
            n = len(shots)
            if n < cfg.diet_min_shots:
                continue
            zones: dict[str, dict[str, int]] = {}
            for s in shots:
                z = zones.setdefault(str(_get(s, "zone")), {"fga": 0, "fgm": 0})
                z["fga"] += 1
                z["fgm"] += 1 if _get(s, "made") else 0
            mid = zones.get("midrange", {"fga": 0, "fgm": 0})
            share = mid["fga"] / n
            if share < cfg.diet_midrange_share:
                continue
            ev = {"n": n, "n_unit": "shots", "by_zone": zones, "midrange_share": share,
                  "nba_pts_per_shot": {"at_rim": 1.345, "long_mid": 0.824, "corner_three": 1.161}}
            text = f"{{team}} {mid['fga']} of {n} shots from midrange, {mid['fgm']} made"
            cards.append(_Card("shot_diet", team.value, ("shot_diet", team.value, n // 5), text, ev, team=team))
        return cards

    def _oreb(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        cards = []
        for team in _TEAMS:
            line, opp = _team_line(stats, team), _team_line(stats, team.other())
            if line is None or opp is None:
                continue
            oreb, opp_dreb = int(_num(_get(line, "oreb"))), int(_num(_get(opp, "dreb")))
            chances = oreb + opp_dreb
            if chances <= 0 or chances < cfg.oreb_min_chances or oreb / chances < cfg.oreb_pct_high:
                continue
            rate = oreb / chances
            ev = {"n": chances, "n_unit": "rebound chances", "oreb": oreb, "opp_dreb": opp_dreb, "oreb_pct": rate,
                  "second_chance_pts": _get(line, "second_chance_pts"), "nba_orb_pct": 0.260}
            text = f"{{team}} {oreb} of {chances} offensive rebound chances ({_pct(rate)}%)"
            cards.append(_Card("oreb", team.value, ("oreb", team.value, chances // 5), text, ev, team=team))
        return cards

    def _turnovers(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        events = sorted(self._events.values(), key=lambda e: e.t)
        cards = []
        for team in _TEAMS:
            tos = [e for e in events if e.kind is EventKind.TURNOVER and self._team_of_player(e.player_id, e.team) is team]
            if len(tos) < cfg.tov_burst_count:
                continue
            starts = [e.t for e in events if e.kind is EventKind.POSSESSION
                      and (_team(e.team) or self._team_of_player(e.player_id)) is team]
            window = starts[-cfg.tov_burst_possessions:]
            recent = [e for e in tos if window and e.t >= window[0]]
            if window and len(recent) <= len(window):
                n, unit, tail = len(window), "possessions", f"in last {len(window)} possessions"
            else:  # no POSSESSION events, or too few to hold these turnovers (each TO ends one)
                recent = [e for e in tos if e.t >= clock.t - cfg.tov_burst_window_s]
                minutes = int(round(cfg.tov_burst_window_s / 60.0))
                n, unit, tail = len(recent), "turnovers", f"in last {minutes} minutes"
            if len(recent) < cfg.tov_burst_count:
                continue
            types = Counter(str(e.attrs.get("type", "other")) for e in recent)
            ev = {"n": n, "n_unit": unit, "turnovers": len(recent), "types": dict(types),
                  "event_ids": [e.event_id for e in recent]}
            text = f"{{team}} {len(recent)} turnovers {tail}"
            cards.append(_Card("turnovers", team.value, ("turnovers", team.value, recent[-1].event_id), text, ev,
                               team=team))
        return cards

    def _lineups(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        lineups = dict(_get(stats, "lineups") or {})
        flagged = {k: v for k, v in lineups.items() if _get(v, "current") or _get(v, "active")}
        cards = []
        for lid, lu in (flagged or lineups).items():
            poss = next((_num(_get(lu, k)) for k in ("possessions", "poss", "n_poss") if _get(lu, k) is not None), None)
            if poss is None or poss < cfg.lineup_min_possessions:
                continue  # no possession count, no card (narration.md §7.2)
            pm = _get(lu, "plus_minus")
            net = _num(pm) if pm is not None else _num(_get(lu, "pts_for")) - _num(_get(lu, "pts_against"))
            if abs(net) < cfg.lineup_min_abs_net:
                continue
            team = _team(_get(lu, "team"))
            what = "current five" if flagged else "lineup"
            ev = {"n": int(poss), "n_unit": "possessions", "plus_minus": net, "players": list(_get(lu, "players") or []),
                  "seconds": _get(lu, "seconds"), "lineup_id": str(lid)}
            text = f"{{team}} {what} {int(net):+d} over {int(poss)} possessions"
            cards.append(_Card("lineup", str(lid), ("lineup", str(lid), int(poss) // 5, int(net) > 0), text, ev,
                               team=team))
        return cards

    def _fatigue(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        cards = []
        for pid, tops in self._speed_tops.items():
            if len(tops) < cfg.fatigue_min_bins:
                continue
            earlier, recent = tops[: -cfg.fatigue_recent_bins], tops[-cfg.fatigue_recent_bins:]
            baseline = float(np.percentile(earlier, 75))
            if baseline < cfg.fatigue_min_baseline_fts:
                continue
            drop = 1.0 - max(recent) / baseline
            if drop < cfg.fatigue_drop:
                continue
            unit = "minutes" if cfg.fatigue_bin_s == 60.0 else "bins"
            ev = {"n": len(tops), "n_unit": unit, "baseline_top_fts": baseline, "recent_top_fts": max(recent),
                  "drop": drop}
            text = f"{{player}} top speed down {_pct(drop)}% vs earlier"
            cards.append(_Card("fatigue", pid, ("fatigue", pid, int(_pct(drop) // 5)), text, ev,
                               team=self._team_of_player(pid), player_id=pid))
        return cards

    def _decisions(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        by_player: dict[str, list[Any]] = {}
        # "last N" needs time order; the contract does not promise the list is sorted (stats-core
        # keeps first-grading order, and a late regrade or an out-of-order event breaks it)
        for d in sorted(_get(stats, "decisions") or [], key=lambda d: _num(_get(d, "t"))):
            pid = _get(d, "player_id")
            if pid is not None:
                by_player.setdefault(str(pid), []).append(d)
        cards = []
        for pid, decs in by_player.items():
            last = decs[-cfg.decision_window:]
            poor = [d for d in last if _get(d, "label") == "poor"]
            if len(poor) < cfg.decision_min_poor:
                continue
            best_kinds = Counter(str(_get(_get(d, "best"), "kind")) for d in poor)
            ev = {"n": len(last), "n_unit": "decisions", "poor": len(poor),
                  "mean_poor_delta": float(np.mean([_num(_get(d, "delta")) for d in poor])),
                  "best_alternatives": dict(best_kinds)}
            text = f"{{player}} {len(poor)} low-value choices in last {len(last)} graded"
            key = ("decision", pid, _get(poor[-1], "event_id") or _get(poor[-1], "t"))
            cards.append(_Card("decision", pid, key, text, ev, team=self._team_of_player(pid), player_id=pid))
        return cards

    def _ft_rate(self, stats: Any, clock: _Clock) -> list[_Card]:
        cfg = self.config
        cards = []
        for team in _TEAMS:
            line = _team_line(stats, team)
            fga, fta = int(_num(_get(line, "fga"))), int(_num(_get(line, "fta")))
            if line is None or fga < cfg.ft_rate_min_fga or fta / fga < cfg.ft_rate_high:
                continue
            rate = fta / fga
            ev = {"n": fga, "n_unit": "shots", "fta": fta, "fga": fga, "ft_rate": rate, "nba_ft_rate": 0.264}
            text = f"{{team}} FT rate {_pct(rate)}%: {fta} FTA on {fga} FGA"
            cards.append(_Card("ft_rate", team.value, ("ft_rate", team.value, fga // 5), text, ev, team=team))
        return cards

    def _team_fouls(self, stats: Any, clock: _Clock) -> list[_Card]:
        if self.rules is None or clock.period is None:
            return []
        if (self.rules.bonus or {}).get("two_shots_from") is None:
            return []  # a custom RuleSpec without a penalty threshold: no bonus card (not a crash per update)
        excluded = set(self.config.team_foul_excluded_types)
        fouls = [e for e in self._events.values() if e.kind is EventKind.FOUL
                 and str(e.attrs.get("type", e.attrs.get("foul_type", ""))) not in excluded]
        if not fouls:
            return []
        label = self._period_label(clock.period)
        cards = []
        for team in _TEAMS:
            mine = [self._event_ctx.get(e.event_id, (None, None)) for e in fouls
                    if self._team_of_player(e.player_id, e.team) is team]
            mine = [(p, gc) for p, gc in mine if p is not None]
            st = penalty_state(self.rules, clock.period, clock.game_clock, mine)
            if st.award == "none":
                continue
            if st.award == "one_and_one":
                tail = "opponent in one-and-one" if st.in_bonus else "next is one-and-one"
            else:
                tail = "opponent in the bonus" if st.fouls >= st.two_shots_from else "next is two shots"
            ev = {"n": st.fouls, "n_unit": "team fouls", "award_next": st.award, "in_bonus": st.in_bonus,
                  "two_shots_from": st.two_shots_from, "one_and_one_from": st.one_and_one_from,
                  "periods": list(st.periods)}
            text = f"{{team}} {st.fouls} team fouls in {label}, {tail}"
            cards.append(_Card("team_fouls", team.value, ("team_fouls", team.value, st.periods, st.fouls), text, ev,
                               team=team))
        return cards

    # ------------------------------------------------------------------ fatigue bins

    def _close_bin(self, pid: str, b: _SpeedBin) -> None:
        cfg = self.config
        if b.speeds and b.last_t - b.first_t >= cfg.fatigue_min_coverage_s:
            self._speed_tops.setdefault(pid, []).append(float(np.percentile(b.speeds, cfg.fatigue_quantile)))


def _team_keyed(mapping: Mapping, team: Team) -> Any:
    """``mapping[team]`` for Team-keyed (in memory) or "home"/"away"-keyed (``to_dict``) dicts."""
    value = mapping.get(team)
    return value if value is not None else mapping.get(team.value)


def _team_line(stats: Any, team: Team) -> Any:
    return _team_keyed(_get(stats, "teams") or {}, team)


def _opt_int(v: Any) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _opt_float(v: Any) -> float | None:
    try:
        f = None if v is None else float(v)
    except (TypeError, ValueError):
        return None
    return f if f is None or math.isfinite(f) else None
