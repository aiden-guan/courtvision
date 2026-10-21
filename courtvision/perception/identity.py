"""Two-level identity: short-lived tracks -> stable (team, jersey) player ids.

Rules (cv_pipeline.md §3.4, §4.2; RESEARCH.md §2.2):

* A track gets the roster id of (team, voted jersey) once its jersey votes are confirmed among the
  team's roster numbers (``JerseyVoter``). Without a roster the id is ``"<team>-<jersey>"``.
* Otherwise the id is provisional: ``"<team>-t<track_id>"`` (``"unk-t<track_id>"`` with no team).
* Long-lost reacquisition (Basketball-SORT "RLLI"): when a track disappears its identity goes to a
  lost pool; a *new* track of the same team, with a compatible jersey, within a court radius that
  grows with the time lost, takes that identity over (Hungarian on court distance) together with
  the old track's jersey votes. The same track coming back (the tracker kept it through the
  occlusion) takes its own parked identity back first.
* A roster id is never held by two live tracks: claimants are ranked by jersey vote mass plus an
  incumbency bonus, minus a penalty for coasting; losers fall back to provisional ids.
* Same-team contact: two tracks of one team whose boxes overlap may come out of it swapped (same
  colours, so the tracker cannot tell). When the contact ends, each track's earlier jersey reads are
  discounted, so a few fresh reads re-decide who it is (a confirmed id is still kept without fresh
  contrary evidence). A track the tracker re-associates after a long coast gets the same discount.
* At most ``max_per_team`` live tracks keep a team (5 on court); the weakest extras lose it.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from courtvision.types import ObjClass, Roster, Team, Track

from .jersey import JerseyVoter

__all__ = ["IdentityConfig", "IdentityManager", "provisional_id"]

log = logging.getLogger(__name__)

IdentityTuple = tuple[str, Team | None, str | None]


@dataclass
class IdentityConfig:
    """Identity-layer settings."""

    min_votes: int = 3  # jersey reads before confirmation; sourced (Roboflow 3-read rule)
    min_share: float = 0.6  # vote share of the leading number; tuned
    single_digit_weight: float = 0.5  # Koshkina 1-digit down-weighting; sourced (factor tuned)
    jersey_window: int | None = 12  # newest reads kept per track (~2 s at one read per 5 frames); tuned
    # 3 confident identical reads in a row also confirm (Roboflow), bypassing min_share. Off: the spec
    # rule is >= 3 reads AND share >= 0.6; on 12 sim windows (3 seeds) roster-id accuracy was identical
    # either way, and the streak lets 3 weak or partial misreads flip a confirmed number; tuned.
    jersey_streak: bool = False
    max_per_team: int | None = 5  # players on court per team (None for drills); sourced (rules)
    relink_max_lost_s: float = 5.0  # how long a lost identity can be reacquired; tuned (paper gives no B)
    relink_radius_ft: float = 8.5  # Basketball-SORT beta = 260 cm (our reading of Eq. 3; unverified)
    relink_speed_ft_s: float = 12.0  # radius growth per second lost; tuned
    relink_max_radius_ft: float = 30.0  # tuned
    relink_new_track_frames: int = 30  # unidentified tracks younger than this may take a lost id; tuned
    # Inherited jersey reads count this much (vs 1.0 for the track's own reads): a re-link is a
    # proximity guess, and full-weight inherited reads let a wrong re-link outvote the true owner's
    # fresh reads (seen on sim seed 11 after a same-team crossing); tuned.
    relink_vote_weight: float = 0.5
    # Same-team contact: two live tracks of one team whose boxes overlap at least this IoU (a crossing,
    # or one player behind the other in one colour blob) may come out of it swapped, and the tracker
    # cannot tell (same colours). When the contact ends, the reads each track made before it are
    # scaled by contact_discount, so >= min_votes fresh agreeing reads outvote them and a swapped id is
    # corrected in ~0.5 s instead of after the window turns over (~1.3-2 s; sim train seeds: wrong
    # same-team ids are ~93 % of wrong ids, and runs started at a crossing). The same discount applies to
    # a track that comes back after a long occlusion. Tuned on sim train seeds.
    contact_iou: float = 0.1
    contact_discount: float = 0.1
    fps: float = 30.0  # converts frame gaps to seconds when ``t`` is not given; design
    incumbency_bonus: float = 1.0  # vote-mass bonus for the track already holding an id; tuned
    coast_penalty: float = 0.2  # per frame a claimant has not been matched; tuned
    min_team_conf: float = 0.0  # weaker team votes count as "no team"; tuned
    jersey_ids_without_roster: bool = True  # "<team>-<jersey>" ids when no roster is given; design


@dataclass
class _Lost:
    player_id: str
    team: Team | None
    jersey: str | None
    xy: tuple[float, float] | None
    frame_idx: int
    t: float | None
    track_id: int
    roster_bound: bool


def provisional_id(team: Team | None, track_id: int) -> str:
    """``"<team>-t<track_id>"`` or ``"unk-t<track_id>"``."""
    return f"{team.value if team is not None else 'unk'}-t{track_id}"


class IdentityManager:
    """Maps live tracks to player ids, frame by frame (see module docstring)."""

    def __init__(self, roster: Roster | None = None, config: IdentityConfig | None = None) -> None:
        self.roster = roster
        self.config = config or IdentityConfig()
        cfg = self.config
        self.voter = JerseyVoter(cfg.min_votes, cfg.min_share, cfg.single_digit_weight,
                                 window=cfg.jersey_window, streak=cfg.jersey_streak)
        self._bound: dict[int, str] = {}  # track -> non-default id (roster / jersey / relinked)
        self._roster_bound: set[int] = set()  # tracks whose bound id comes from a jersey
        self._prev: dict[int, IdentityTuple] = {}  # last output per live track
        self._last_xy: dict[int, tuple[float, float]] = {}
        self._last_seen: dict[int, tuple[int, float | None]] = {}
        self._lost: dict[str, _Lost] = {}
        self._contact: set[int] = set()  # tracks in a same-team contact last frame

    # ------------------------------------------------------------------ public

    def assign(
        self,
        tracks: list[Track],
        teams: Mapping[int, tuple[Team | None, float]],
        frame_idx: int,
        court_xy: Mapping[int, tuple[float, float]] | None = None,
        t: float | None = None,
    ) -> dict[int, IdentityTuple]:
        """track_id -> (player_id, team, jersey) for the live PLAYER tracks of this frame.

        ``court_xy`` (track -> court ft) and ``t`` enable proximity re-linking; both are optional
        extensions of the ARCHITECTURE signature.
        """
        players = [tr for tr in tracks if tr.cls is ObjClass.PLAYER]
        live = {tr.track_id for tr in players}
        court_xy = court_xy or {}
        team_of = {tr.track_id: self._team(teams, tr.track_id) for tr in players}
        self._update_contacts(players, team_of)  # before this frame's reads: those are fresh evidence
        for tr in players:
            if tr.time_since_update == 0 and tr.jersey is not None:
                self.voter.add(tr.track_id, tr.jersey, tr.jersey_conf)
            if tr.track_id in court_xy:
                self._last_xy[tr.track_id] = court_xy[tr.track_id]
            self._last_seen[tr.track_id] = (frame_idx, t)
        self._retire_lost_tracks(live, frame_idx, t)
        self._expire(frame_idx, t)
        self._restore_returning(players)

        claims = {tr.track_id: self._jersey_claim(tr.track_id, team_of[tr.track_id]) for tr in players}
        self._resolve_claims(players, claims)
        self._relink(players, team_of, frame_idx, t)
        out: dict[int, IdentityTuple] = {}
        for tr in players:
            out[tr.track_id] = self._identity(tr.track_id, team_of[tr.track_id])
        out = self._apply_team_cap(players, out, teams)
        out = self._unique(out, players)
        self._prev = out
        return out

    def lost_identities(self) -> list[str]:
        """Player ids currently waiting in the re-link pool."""
        return sorted(self._lost)

    # ------------------------------------------------------------------ steps

    def _team(self, teams: Mapping[int, tuple[Team | None, float]], track_id: int) -> Team | None:
        team, conf = teams.get(track_id, (None, 0.0))
        return team if team is not None and conf >= self.config.min_team_conf else None

    def _retire_lost_tracks(self, live: set[int], frame_idx: int, t: float | None) -> None:
        for track_id in [k for k in self._prev if k not in live]:
            pid, team, jersey = self._prev.pop(track_id)
            seen_frame, seen_t = self._last_seen.get(track_id, (frame_idx, t))
            self._lost[pid] = _Lost(pid, team, jersey, self._last_xy.get(track_id), seen_frame, seen_t, track_id,
                                    track_id in self._roster_bound)
            self._bound.pop(track_id, None)
            self._roster_bound.discard(track_id)
            self._last_xy.pop(track_id, None)
            self._last_seen.pop(track_id, None)
            log.debug("track %d lost; identity %s parked for re-linking", track_id, pid)

    def _restore_returning(self, players: list[Track]) -> None:
        """A track the tracker kept alive through a long occlusion comes back under its own track id
        (ByteTrack's lost buffer outlives the frames it is reported for). It takes back the identity
        it parked, before ``_relink`` could hand that identity to a new track nearby."""
        parked: dict[int, _Lost] = {}
        for lost in self._lost.values():
            prev = parked.get(lost.track_id)
            if prev is None or lost.frame_idx > prev.frame_idx:
                parked[lost.track_id] = lost
        for tr in players:
            lost = parked.get(tr.track_id)
            if lost is None:
                continue
            for pid in [p for p, lo in self._lost.items() if lo.track_id == tr.track_id]:
                del self._lost[pid]
            if lost.roster_bound or lost.player_id != provisional_id(lost.team, tr.track_id):
                self._bound[tr.track_id] = lost.player_id
                if lost.roster_bound:
                    self._roster_bound.add(tr.track_id)
            self.voter.discount(tr.track_id, self.config.contact_discount)  # re-associated after a long coast
            log.debug("track %d returned; identity %s restored", tr.track_id, lost.player_id)

    def _update_contacts(self, players: list[Track], team_of: dict[int, Team | None]) -> None:
        """Discount the jersey reads of tracks whose same-team contact ended this frame."""
        cfg = self.config
        now: set[int] = set()
        if cfg.contact_iou is not None and cfg.contact_iou > 0:
            for i, a in enumerate(players):
                ta = team_of[a.track_id]
                if ta is None:
                    continue
                for b in players[i + 1:]:
                    if team_of[b.track_id] is ta and a.bbox.iou(b.bbox) >= cfg.contact_iou:
                        now.add(a.track_id)
                        now.add(b.track_id)
        live = {tr.track_id for tr in players}
        for tid in (self._contact - now) & live:
            self.voter.discount(tid, cfg.contact_discount)
            log.debug("track %d: same-team contact ended; older jersey reads discounted", tid)
        self._contact = now

    def _expire(self, frame_idx: int, t: float | None) -> None:
        for pid, lost in list(self._lost.items()):
            if self._lost_seconds(lost, frame_idx, t) > self.config.relink_max_lost_s:
                del self._lost[pid]
                self.voter.forget(lost.track_id)

    def _lost_seconds(self, lost: _Lost, frame_idx: int, t: float | None) -> float:
        if t is not None and lost.t is not None:
            return max(0.0, t - lost.t)
        return max(0, frame_idx - lost.frame_idx) / self.config.fps

    def _allowed(self, team: Team | None) -> set[str] | None:
        if self.roster is None:
            return None
        if team is None:
            return set()
        return {e.jersey for e in self.roster.entries if e.team is team}

    def _jersey_claim(self, track_id: int, team: Team | None) -> str | None:
        """Id implied by the confirmed (team, jersey) of this track, if any."""
        if team is None:
            return None
        number, _share = self.voter.result(track_id, self._allowed(team))
        if number is None:
            return None
        if self.roster is not None:
            entry = self.roster.by_jersey(team, number)
            return entry.player_id if entry is not None else None
        return f"{team.value}-{number}" if self.config.jersey_ids_without_roster else None

    def _resolve_claims(self, players: list[Track], claims: dict[int, str | None]) -> None:
        """Bind confirmed jersey ids, one live track per id."""
        cfg = self.config
        by_id: dict[str, list[Track]] = {}
        for tr in players:
            pid = claims[tr.track_id]
            if pid is None and tr.track_id in self._roster_bound:
                pid = self._bound.get(tr.track_id)  # sticky: keep a confirmed id without fresh evidence
            if pid is not None:
                by_id.setdefault(pid, []).append(tr)
        winners: dict[int, str] = {}
        for pid, claimants in sorted(by_id.items()):
            def strength(tr: Track, pid: str = pid) -> tuple[float, int]:
                jersey = self._jersey_of(pid)
                mass = self.voter.weight(tr.track_id, jersey) if jersey is not None else 0.0
                bonus = cfg.incumbency_bonus if self._bound.get(tr.track_id) == pid else 0.0
                return mass + bonus - cfg.coast_penalty * tr.time_since_update, -tr.track_id
            winner = max(claimants, key=strength)
            winners[winner.track_id] = pid
            if len(claimants) > 1:
                log.debug("id %s claimed by tracks %s; kept by %d",
                          pid, [c.track_id for c in claimants], winner.track_id)
        for tr in players:
            tid = tr.track_id
            if tid in winners:
                if self._bound.get(tid) != winners[tid]:
                    log.info("track %d identified as %s", tid, winners[tid])
                self._bound[tid] = winners[tid]
                self._roster_bound.add(tid)
                self._lost.pop(winners[tid], None)
            elif tid in self._roster_bound:  # lost its id to a stronger claimant
                self._bound.pop(tid, None)
                self._roster_bound.discard(tid)

    def _relink(self, players: list[Track], team_of: dict[int, Team | None], frame_idx: int, t: float | None) -> None:
        """Give lost identities to new, unidentified tracks nearby (Hungarian on court distance)."""
        cfg = self.config
        fresh = [tr for tr in players if tr.track_id not in self._bound and tr.track_id in self._last_xy
                 and tr.age <= cfg.relink_new_track_frames]
        pool = [lost for lost in self._lost.values() if lost.xy is not None]
        if not fresh or not pool:
            return
        cost = np.full((len(fresh), len(pool)), np.inf)
        for i, tr in enumerate(fresh):
            team = team_of[tr.track_id]
            lead, _share, _n = self.voter.best(tr.track_id, self._allowed(team))
            x, y = self._last_xy[tr.track_id]
            for j, lost in enumerate(pool):
                if lost.team is None or lost.team is not team:
                    continue
                if lost.jersey is not None and lead is not None and lead != lost.jersey:
                    continue
                radius = min(cfg.relink_max_radius_ft,
                             cfg.relink_radius_ft + cfg.relink_speed_ft_s * self._lost_seconds(lost, frame_idx, t))
                assert lost.xy is not None
                d = math.hypot(x - lost.xy[0], y - lost.xy[1])
                if d <= radius:
                    cost[i, j] = d
        finite = np.where(np.isfinite(cost), cost, 1e9)
        rows, cols = linear_sum_assignment(finite)
        for r, c in zip(rows, cols):
            if not np.isfinite(cost[r, c]):
                continue
            tr, lost = fresh[r], pool[c]
            self._bound[tr.track_id] = lost.player_id
            if lost.roster_bound:
                self._roster_bound.add(tr.track_id)
            self.voter.merge(lost.track_id, tr.track_id, weight=cfg.relink_vote_weight)
            del self._lost[lost.player_id]
            log.info("track %d re-linked to lost identity %s (%.1f ft)", tr.track_id, lost.player_id, cost[r, c])

    def _identity(self, track_id: int, team: Team | None) -> IdentityTuple:
        pid = self._bound.get(track_id)
        if pid is None:
            number, _ = self.voter.result(track_id, self._allowed(team))
            return provisional_id(team, track_id), team, number
        entry = self.roster.get(pid) if self.roster is not None else None
        if entry is not None:
            return pid, entry.team, entry.jersey
        lost_team = self._team_from_id(pid)
        return pid, team if team is not None else lost_team, self._jersey_of(pid)

    def _apply_team_cap(
        self, players: list[Track], out: dict[int, IdentityTuple], teams: Mapping[int, tuple[Team | None, float]]
    ) -> dict[int, IdentityTuple]:
        cap = self.config.max_per_team
        if cap is None:
            return out
        by_track = {tr.track_id: tr for tr in players}
        for team in (Team.HOME, Team.AWAY):
            members = [tid for tid, (_, tm, _) in out.items() if tm is team]
            if len(members) <= cap:
                continue

            def rank(tid: int) -> tuple[int, int, float, int, int]:
                tr = by_track[tid]
                identified = int(tid in self._roster_bound)
                return (identified, -tr.time_since_update, teams.get(tid, (None, 0.0))[1], tr.hits, -tid)

            ordered = sorted(members, key=rank, reverse=True)
            for tid in ordered[cap:]:
                pid, _, jersey = out[tid]
                if tid in self._roster_bound:
                    continue  # never strip a jersey-confirmed player; the cap applies to the unidentified
                log.debug("team cap: track %d demoted from %s", tid, team.value)
                out[tid] = (provisional_id(None, tid) if tid not in self._bound else pid, None, jersey)
        return out

    def _unique(self, out: dict[int, IdentityTuple], players: list[Track]) -> dict[int, IdentityTuple]:
        """Invariant: one player id per live track. On a collision (a track that reappears after
        handing its provisional id over) the bound / longer-lived track keeps the id and the other
        is left out of this frame's output."""
        hits = {tr.track_id: tr.hits for tr in players}
        holder: dict[str, int] = {}
        for tid in sorted(out, key=lambda k: (k not in self._bound, -hits.get(k, 0), k)):
            pid = out[tid][0]
            if pid in holder:
                log.debug("id %s already held by track %d; track %d left out", pid, holder[pid], tid)
                continue
            holder[pid] = tid
        return {tid: out[tid] for tid in out if holder.get(out[tid][0]) == tid}

    # ------------------------------------------------------------------ id helpers

    def _jersey_of(self, pid: str) -> str | None:
        if self.roster is not None:
            entry = self.roster.get(pid)
            if entry is not None:
                return entry.jersey
        head, _, tail = pid.partition("-")
        if head in ("home", "away") and tail and not tail.startswith("t"):
            return tail
        return None

    @staticmethod
    def _team_from_id(pid: str) -> Team | None:
        head = pid.partition("-")[0]
        return Team(head) if head in ("home", "away") else None
