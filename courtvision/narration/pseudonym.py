"""Per-game player pseudonyms (D16, RESEARCH.md §7A.7) and identity-tier rendering (§4.1).

Cloud payloads (Claude FactPackets, Jev states) carry opaque per-game tokens ``P01``..``P30``,
never names. The map lives in RAM for one game. Only after the deterministic validator has passed
does :meth:`PseudonymMap.render` replace ``{P07}`` with a name, "Home's number 23", or the team name,
according to the identity tiers:

* identity confidence >= 0.95                      -> roster name
* 0.80 <= identity < 0.95 and jersey OCR >= 0.95   -> "<Team>'s number <jersey>"
* otherwise                                        -> the team name (Mori et al.'s team fallback)
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Mapping
from dataclasses import dataclass

from courtvision.types import Roster, RosterEntry, Team

log = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"\{(P\d{2,3})\}")


@dataclass
class PseudonymConfig:
    """Token pool and identity-tier thresholds."""

    pool_size: int = 30  # tokens P01..P30 per game (RESEARCH.md §7A.7); grows if a game needs more; sourced
    name_conf: float = 0.95  # ID >= 0.95 -> roster name (RESEARCH.md §4.1, narration.md §3.4); unverified
    jersey_id_conf: float = 0.80  # 0.80 <= ID < 0.95 -> "Team's number N" (same sources); unverified
    jersey_ocr_conf: float = 0.95  # ...only when the jersey OCR itself is >= 0.95 (same sources); unverified
    unknown_player_text: str = "a player"  # token with no player or team (should not pass the validator)


def possessive(name: str) -> str:
    """"Home" -> "Home's", "Hawks" -> "Hawks'"."""
    return f"{name}'" if name.endswith("s") else f"{name}'s"


def team_from_player_id(player_id: str) -> Team | None:
    """Team implied by a provisional id ("home-t12", "away-23"); None for "unk-t5"."""
    head = player_id.split("-", 1)[0].lower()
    return Team(head) if head in (Team.HOME.value, Team.AWAY.value) else None


class PseudonymMap:
    """Per-game player <-> token map. Tokens are shuffled (not roster or jersey order)."""

    def __init__(self, roster: Roster, seed: int = 0, config: PseudonymConfig | None = None) -> None:
        self.config = config or PseudonymConfig()
        n = max(self.config.pool_size, len(roster.entries))
        numbers = list(range(1, n + 1))
        random.Random(seed).shuffle(numbers)
        self._free: list[int] = numbers
        self._next_extra = n + 1
        self._by_player: dict[str, str] = {}
        self._by_token: dict[str, str] = {}
        for entry in roster.entries:
            self._assign(entry.player_id)

    # ------------------------------------------------------------------ mapping

    def _assign(self, player_id: str) -> str:
        if self._free:
            num = self._free.pop(0)
        else:
            num = self._next_extra
            self._next_extra += 1
            log.warning("pseudonym pool exhausted; issuing P%02d", num)
        tok = f"P{num:02d}"
        self._by_player[player_id] = tok
        self._by_token[tok] = player_id
        return tok

    def token(self, player_id: str) -> str:
        """Token for ``player_id`` ("P07"); provisional ids get the next free token on first use."""
        tok = self._by_player.get(player_id)
        return tok if tok is not None else self._assign(player_id)

    def player(self, token: str) -> str | None:
        """Player id behind a token ("P07" or "{P07}"); None if unknown."""
        return self._by_token.get(token.strip("{}"))

    def tokens(self) -> dict[str, str]:
        """Copy of token -> player id (RAM only; never send it anywhere)."""
        return dict(self._by_token)

    def __contains__(self, player_id: object) -> bool:
        return player_id in self._by_player

    # ------------------------------------------------------------------ rendering

    def display(self, token: str, roster: Roster, id_conf: Mapping[str, float] | None = None,
                jersey_conf: Mapping[str, float] | None = None) -> str:
        """Spoken form of one token under the identity tiers (missing confidences count as 1.0)."""
        pid = self.player(token)
        if pid is None:
            log.warning("render: unknown token %s", token)
            return self.config.unknown_player_text
        tok = token.strip("{}")
        entry = roster.get(pid)
        team = entry.team if entry else team_from_player_id(pid)
        idc = _lookup(id_conf, pid, tok)
        jc = _lookup(jersey_conf, pid, tok)
        cfg = self.config
        if entry is not None and idc >= cfg.name_conf:
            return entry.name
        if entry is not None and entry.jersey and cfg.jersey_id_conf <= idc and jc >= cfg.jersey_ocr_conf:
            return f"{possessive(_team_name(roster, entry.team))} number {entry.jersey}"
        if team is not None:
            return _team_name(roster, team)
        return cfg.unknown_player_text

    def render(self, text: str, roster: Roster, id_conf: dict[str, float] | None = None,
               jersey_conf: dict[str, float] | None = None) -> str:
        """Replace every ``{Pnn}`` with its tiered display form. Confidence dicts may be keyed by
        player id or token. Call only after the grounding validator passed."""

        def repl(m: re.Match[str]) -> str:
            shown = self.display(m.group(1), roster, id_conf, jersey_conf)
            if shown and shown[0].islower() and _sentence_start(text, m.start()):
                shown = shown[0].upper() + shown[1:]
            return shown

        return TOKEN_RE.sub(repl, text)

    def pseudonymise(self, text: str, roster: Roster) -> str:
        """Replace roster names in ``text`` with ``{Pnn}`` (full names, then unique surnames)."""
        out = text
        entries = sorted(roster.entries, key=lambda e: -len(e.name))
        for e in entries:
            if e.name:
                out = re.sub(rf"(?<!\w){re.escape(e.name)}(?!\w)", "{" + self.token(e.player_id) + "}", out)
        for e, surname in _unique_surnames(entries):
            out = re.sub(rf"(?<![\w{{]){re.escape(surname)}(?!\w)", "{" + self.token(e.player_id) + "}", out)
        return out


# --------------------------------------------------------------------------- shared default map

# One map per (roster object, seed): every caller that does not hold its own map (build_facts without
# ``pseudo``, the renderer) must see the SAME tokens, including the lazily issued provisional ones.
_DEFAULT_MAPS: dict[tuple[int, int], tuple[Roster, PseudonymMap]] = {}
_DEFAULT_MAPS_MAX = 8  # games kept in RAM at once (one per live roster); older ones are dropped


def pseudonyms_for(roster: Roster, seed: int = 0) -> PseudonymMap:
    """The shared per-game map for ``roster`` (same object -> same map, so tokens stay stable)."""
    key = (id(roster), seed)
    hit = _DEFAULT_MAPS.get(key)
    if hit is not None and hit[0] is roster:
        return hit[1]
    pm = PseudonymMap(roster, seed=seed)
    if len(_DEFAULT_MAPS) >= _DEFAULT_MAPS_MAX:
        _DEFAULT_MAPS.pop(next(iter(_DEFAULT_MAPS)))
    _DEFAULT_MAPS[key] = (roster, pm)  # holding the roster keeps id() from being reused while cached
    return pm


# --------------------------------------------------------------------------- helpers


def _lookup(conf: Mapping[str, float] | None, pid: str, tok: str) -> float:
    if not conf:
        return 1.0
    if pid in conf:
        return float(conf[pid])
    return float(conf.get(tok, 1.0))


def _team_name(roster: Roster, team: Team) -> str:
    return roster.team_names.get(team, team.value.capitalize())


def _sentence_start(text: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and text[j] in " \t\"'(":
        j -= 1
    return j < 0 or text[j] in ".!?:"


def _unique_surnames(entries: list[RosterEntry]) -> list[tuple[RosterEntry, str]]:
    counts: dict[str, int] = {}
    for e in entries:
        parts = e.name.split()
        if len(parts) >= 2:
            counts[parts[-1]] = counts.get(parts[-1], 0) + 1
    out = []
    for e in entries:
        parts = e.name.split()
        if len(parts) >= 2 and counts[parts[-1]] == 1 and len(parts[-1]) >= 3:
            out.append((e, parts[-1]))
    return out
