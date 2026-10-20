"""Clock, period, timeout, possession and foul rules, separate from geometry (D15, RESEARCH §7A.6).

Each ruleset is a YAML file in ``courtvision/config/rules/<ruleset>.yaml`` whose keys are exactly
the ``RuleSpec`` fields; values and their sources are documented in the YAML comments. State or
tournament overrides (``config/rules/overrides/<name>.yaml`` or an explicit path) are deep-merged
on top in order: nested dicts merge key by key, every other value (lists, scalars, ``null``)
replaces. Unknown keys, at the top level or inside the documented dict fields, raise ValueError.

Dict field schemas (keys allowed in YAML; see ``config/rules/overrides/README.md``):

* ``running_clock``: ``stop_last_s``, ``stop_scope`` ("period" | "half" | "game"),
  ``mercy_margin``, ``mercy_from_period``, ``resume_below_margin``.
* ``timeouts`` (required ``scope``, ``length_s``, ``per_ot``): counts per game or per half, lengths,
  late-game limits, carry-over, optional ``media`` format.
* ``bonus`` (required ``scope``, ``two_shots_from``): team-foul thresholds (the foul number that
  first awards the penalty), late-period and OT variants.
* ``tech_ft`` (required ``n_ft``, ``shooter``, ``resume``): technical-foul free throws, plus
  flagrant/disruptive variants.
* ``clock_stops_after_fg``: ``q1_3_last_s`` (every regulation period except the last) and
  ``q4_ot_last_s`` (the last regulation period and every OT, also for halves formats). Use
  ``RuleSpec.clock_stops_after_fg_at`` rather than reading the keys.
* ``intervals``: ``between_periods_s``, ``halftime_s``, ``before_ot_s``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = ["RULES_DIR", "OVERRIDES_DIR", "RuleSpec", "available_overrides", "available_rulesets", "rules_for"]

log = logging.getLogger(__name__)

RULES_DIR = Path(__file__).resolve().parent.parent / "config" / "rules"
OVERRIDES_DIR = RULES_DIR / "overrides"


@dataclass(frozen=True)
class RuleSpec:
    """Game rules for one ruleset; load with ``rules_for``."""

    ruleset: str  # "nba" | "fiba" | "ncaa_m" | "ncaa_w" | "nfhs" | custom id
    period_s: int
    n_periods: int
    ot_s: int
    halves: bool  # NCAA men and many AAU / rec formats
    running_clock: dict | None  # e.g. {"stop_last_s": 120} or {"mercy_margin": 35}
    timeouts: dict  # counts per half/game/OT, lengths, carry-over, late-game limits
    possession: str  # "arrow" | "jump" (held balls and period starts)
    bonus: dict
    foul_out: int
    tech_ft: dict  # {"n_ft": 1|2, "shooter": "any", "resume": "poi"|"division_line", ...}
    shot_clock_s: int | None  # None = no shot clock (many NFHS states)
    shot_clock_reset_s: int | None  # offensive rebound after rim contact; None = no shot clock
    clock_stops_after_fg: dict | None  # None = never (NFHS)
    # Additions beyond the ARCHITECTURE field list (defaults keep positional construction valid).
    foul_out_counts_technicals: bool = False  # NFHS counts personal + technical fouls together
    backcourt_s: int | None = None  # backcourt (8/10-second) count
    intervals: dict = field(default_factory=dict)  # break lengths for dead-ball windows

    @property
    def has_shot_clock(self) -> bool:
        return self.shot_clock_s is not None

    @property
    def regulation_s(self) -> int:
        return self.period_s * self.n_periods

    def is_overtime(self, period: int) -> bool:
        return period > self.n_periods

    def period_length_s(self, period: int) -> int:
        """Length of ``period`` (1-based; periods after ``n_periods`` are overtimes)."""
        if period < 1:
            raise ValueError("periods are numbered from 1")
        return self.ot_s if self.is_overtime(period) else self.period_s

    def clock_stops_after_fg_at(self, period: int, remaining_s: float) -> bool:
        """Does the game clock stop after a made field goal at this period / remaining time?"""
        rule = self.clock_stops_after_fg
        if not rule:
            return False
        key = "q4_ot_last_s" if period >= self.n_periods else "q1_3_last_s"
        limit = rule.get(key)
        return limit is not None and remaining_s <= float(limit)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------- schema

_REQUIRED_SUBKEYS: dict[str, frozenset[str]] = {
    "timeouts": frozenset({"scope", "length_s", "per_ot"}),
    "bonus": frozenset({"scope", "two_shots_from"}),
    "tech_ft": frozenset({"n_ft", "shooter", "resume"}),
}
_TIMEOUT_KEYS = frozenset({
    "scope", "count", "length_s", "short_count", "short_length_s", "mandatory_length_s", "first_half",
    "second_half", "max_in_q4", "max_late_q4", "late_q4_s", "per_ot", "per_ot_length_s", "carry_over",
    "carry_to_second_half_max", "media",
})
_ALLOWED_SUBKEYS: dict[str, frozenset[str]] = {
    "running_clock": frozenset({"stop_last_s", "stop_scope", "mercy_margin", "mercy_from_period", "resume_below_margin"}),
    "timeouts": _TIMEOUT_KEYS,
    "bonus": frozenset({"scope", "one_and_one_from", "two_shots_from", "late_period_s", "late_two_shots_from",
                        "ot_two_shots_from", "ot_counts_toward_last_period"}),
    "tech_ft": frozenset({"n_ft", "shooter", "resume", "flagrant_n_ft", "flagrant_shooter", "flagrant_resume",
                          "flagrant_disqualifies"}),
    "clock_stops_after_fg": frozenset({"q1_3_last_s", "q4_ot_last_s"}),
    "intervals": frozenset({"between_periods_s", "halftime_s", "before_ot_s"}),
}
_MEDIA_KEYS = _TIMEOUT_KEYS - {"media"}  # timeouts.media: the media-format allotment
_CHOICES: dict[tuple[str, str], frozenset[str]] = {
    ("timeouts", "scope"): frozenset({"game", "half"}),
    ("bonus", "scope"): frozenset({"period", "half"}),
    ("running_clock", "stop_scope"): frozenset({"period", "half", "game"}),
    ("tech_ft", "shooter"): frozenset({"any", "fouled"}),
    ("tech_ft", "flagrant_shooter"): frozenset({"any", "fouled"}),
    ("tech_ft", "resume"): frozenset({"poi", "division_line", "frontcourt", "sideline", "spot"}),
    ("tech_ft", "flagrant_resume"): frozenset({"poi", "division_line", "frontcourt", "sideline", "spot"}),
}
_POS_INTS = ("period_s", "n_periods", "ot_s", "foul_out")
_OPT_POS_INTS = ("shot_clock_s", "shot_clock_reset_s", "backcourt_s")
_DICTS = ("timeouts", "bonus", "tech_ft", "intervals")
_OPT_DICTS = ("running_clock", "clock_stops_after_fg")
_BOOLS = ("halves", "foul_out_counts_technicals")


def _field_names() -> set[str]:
    return {f.name for f in fields(RuleSpec)}


def _required_fields() -> set[str]:
    return {f.name for f in fields(RuleSpec) if f.default is MISSING and f.default_factory is MISSING}


# ---------------------------------------------------------------------------- loading


def available_rulesets() -> list[str]:
    return sorted(p.stem for p in RULES_DIR.glob("*.yaml"))


def available_overrides() -> list[str]:
    return sorted(p.stem for p in OVERRIDES_DIR.glob("*.yaml"))


def rules_for(ruleset: str, overrides: list[str] | None = None) -> RuleSpec:
    """Load ``config/rules/<ruleset>.yaml`` and deep-merge ``overrides`` in order.

    An override is a name in ``config/rules/overrides/`` (``"no_shot_clock"``) or a path to a
    YAML file. An override may set ``ruleset`` to name a custom ruleset (e.g. an AAU event).
    Raises ValueError for an unknown ruleset, unknown keys or invalid values, and
    FileNotFoundError for a missing override file.
    """
    if isinstance(overrides, (str, Path)):  # a bare string would otherwise iterate per character
        overrides = [overrides]
    base_path = _resolve_base(ruleset)
    data = _load_yaml(base_path)
    for name in overrides or []:
        path = _resolve_override(name)
        data = _deep_merge(data, _load_yaml(path))
        log.info("rules %s: applied override %s", ruleset, path)
    return _build(data, f"{base_path.name} + {list(overrides or [])}")


def _resolve_base(ruleset: str) -> Path:
    key = str(ruleset).strip().lower()
    path = RULES_DIR / f"{key}.yaml"
    if not key or "/" in key or not path.is_file():
        raise ValueError(f"unknown ruleset {ruleset!r}; available: {', '.join(available_rulesets())}")
    return path


def _resolve_override(name: str | Path) -> Path:
    text = str(name)
    candidate = Path(text).expanduser()
    if candidate.suffix in (".yaml", ".yml") or len(candidate.parts) > 1:
        if candidate.is_file():
            return candidate
        bundled = OVERRIDES_DIR / candidate.name  # "no_shot_clock.yaml" names a bundled override
        if len(candidate.parts) == 1 and bundled.is_file():
            return bundled
        raise FileNotFoundError(f"rules override file not found: {candidate}")
    path = OVERRIDES_DIR / f"{text}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"rules override {text!r} not found in {OVERRIDES_DIR}; available: {', '.join(available_overrides())}"
        )
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    _check_keys(data, path)
    return data


def _check_keys(data: dict[str, Any], path: Path) -> None:
    unknown = sorted(set(data) - _field_names())
    if unknown:
        raise ValueError(f"{path}: unknown rule key(s) {unknown}; allowed: {sorted(_field_names())}")
    for name, allowed in _ALLOWED_SUBKEYS.items():
        value = data.get(name)
        if isinstance(value, dict):
            _check_subkeys(value, allowed, f"{path}: {name}")
    timeouts = data.get("timeouts")
    if isinstance(timeouts, dict) and isinstance(timeouts.get("media"), dict):
        _check_subkeys(timeouts["media"], _MEDIA_KEYS, f"{path}: timeouts.media")


def _check_subkeys(value: dict[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ---------------------------------------------------------------------------- validation


def _build(data: dict[str, Any], source: str) -> RuleSpec:
    missing = sorted(_required_fields() - set(data))
    if missing:
        raise ValueError(f"{source}: missing rule key(s) {missing}")
    _validate(data, source)
    return RuleSpec(**data)


def _validate(d: dict[str, Any], source: str) -> None:
    def fail(msg: str) -> None:
        raise ValueError(f"{source}: {msg}")

    if not isinstance(d["ruleset"], str) or not d["ruleset"]:
        fail("ruleset must be a non-empty string")
    for name in _POS_INTS:
        if not _is_int(d[name]) or d[name] <= 0:
            fail(f"{name} must be a positive integer, got {d[name]!r}")
    for name in _OPT_POS_INTS:
        if d.get(name) is not None and (not _is_int(d[name]) or d[name] <= 0):
            fail(f"{name} must be a positive integer or null, got {d[name]!r}")
    for name in _BOOLS:
        if name in d and not isinstance(d[name], bool):
            fail(f"{name} must be true/false, got {d[name]!r}")
    for name in _DICTS:
        if name in d and not isinstance(d[name], dict):
            fail(f"{name} must be a mapping, got {d[name]!r}")
    for name in _OPT_DICTS:
        if d[name] is not None and not isinstance(d[name], dict):
            fail(f"{name} must be a mapping or null, got {d[name]!r}")
    _validate_semantics(d, fail)


def _validate_semantics(d: dict[str, Any], fail: Any) -> None:
    if d["possession"] not in ("arrow", "jump"):
        fail(f"possession must be 'arrow' or 'jump', got {d['possession']!r}")
    if d["halves"] and d["n_periods"] != 2:
        fail(f"halves=true needs n_periods=2, got {d['n_periods']}")
    if d["shot_clock_s"] is None and d["shot_clock_reset_s"] is not None:
        fail("shot_clock_reset_s must be null when shot_clock_s is null")
    if d["shot_clock_s"] is not None and d["shot_clock_reset_s"] is not None and d["shot_clock_reset_s"] > d["shot_clock_s"]:
        fail("shot_clock_reset_s cannot exceed shot_clock_s")
    for name, req in _REQUIRED_SUBKEYS.items():
        missing = sorted(req - set(d[name]))
        if missing:
            fail(f"{name} is missing key(s) {missing}")
    for (name, key), choices in _CHOICES.items():
        value = (d.get(name) or {}).get(key)
        if value is not None and value not in choices:
            fail(f"{name}.{key} must be one of {sorted(choices)}, got {value!r}")


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)
