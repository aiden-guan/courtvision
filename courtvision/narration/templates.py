"""Deterministic narration templates: the always-correct fallback (narration.md §3.4, §9.2; RESEARCH.md §2.6).

``render(events, facts, channel, rng, hedge)`` returns one line in ``{Pnn}`` token form, built only
from the FactPacket made by :func:`courtvision.narration.facts.build_facts`, so it passes the
grounding validator by construction. Channels:

* ``play_by_play``: >= 8 variants per call family (made 2 / made 3 / layup / dunk / free throws /
  miss / block / steal / turnover / rebounds / assist add-on / release call / possession /
  bring-up / foul / timeout / substitution / period start and end / score and run updates);
* hedge variants (``hedge=True`` or a hedge-only packet) never state an outcome or point value;
* ``coach``: flat, numeric cards of <= 12 words;
* ``accessibility``: fixed audio-description lexicon, <= 6 words (+ the confirmed score line);
* ``correction``: "Correction: ..." lines of <= 12 words;
* ``color``: deterministic colour from code-made ``derived`` phrases, the run or the score.

Variety is rng-seeded; a variant whose line is too similar to ``facts["recent_lines"]`` is never
chosen, and :class:`TemplateRenderer` also keeps the last 3 variants per family out (LRU).
"""

from __future__ import annotations

import logging
import random
import re
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from courtvision.narration.validator import OUTCOME_RE, TOKEN_RE, count_words, int_to_words, jaccard, split_sentences
from courtvision.types import Event

log = logging.getLogger(__name__)

CHANNELS = ("play_by_play", "color", "coach", "accessibility", "correction")

# --------------------------------------------------------------------------- play-by-play bank

PBP: dict[str, tuple[str, ...]] = {
    "made3": (
        "{actor} buries the three!", "{actor} drains it from deep!", "{actor} knocks down the three.",
        "Three-ball is good for {actor}!", "{actor} hits from downtown!", "{actor} cashes the three.",
        "Splash! {actor} from deep.", "{actor} connects from beyond the arc.",
        "{actor} from the {zone}... got it!", "Bang! {actor} for three!",
    ),
    "made2": (
        "{actor} knocks it down.", "{actor} hits the {jumper}.", "{actor} with the bucket.",
        "{actor} gets it to go.", "Good from {actor}.", "{actor} drops it in.",
        "{actor} scores from the {zone}.", "Two for {actor}.", "{actor} finds the bottom of the net.",
    ),
    "layup": (
        "{actor} lays it in.", "{actor} finishes at the rim.", "{actor} with the layup.",
        "Easy two for {actor}.", "{actor} gets the layup to drop.", "{actor} drives and scores.",
        "Layup good, {actor}.", "{actor} converts the layup.", "{actor} gets to the rim and finishes.",
    ),
    "dunk": (
        "{actor} throws it down!", "{actor} with the slam!", "{actor} hammers it home!", "Dunk by {actor}!",
        "{actor} flushes it!", "{actor} rocks the rim!", "Up and in! {actor} with the jam.",
        "{actor} goes up and slams it home!", "What a finish from {actor}!",
    ),
    "ft_made": (
        "{actor} hits the free throw.", "Good from the line.", "{actor} makes it.", "Free throw is good.",
        "{actor} converts.", "That one's good.", "Knocked down by {actor}.", "{actor} drops it in.",
        "Swish from the line.",
    ),
    "ft_miss": (
        "{actor} misses the free throw.", "No good from the line.", "Free throw won't fall.",
        "{actor} can't get it down.", "That one's off.", "{actor} can't convert.", "{actor} misses.", "No good.",
        "Not good for {actor}.",
    ),
    "miss": (
        "{actor} misses.", "No good from {actor}.", "{actor} can't connect.", "It won't fall.",
        "{actor} comes up empty.", "{actor} can't hit it.", "Doesn't fall for {actor}.",
        "{actor}'s shot is off.", "{actor} misfires.",
    ),
    "block": (
        "Blocked by {actor}!", "{actor} swats it away!", "Rejected! {actor} with the block.",
        "{actor} gets a piece of it!", "{actor} stuffs {other_tok}!", "{actor} sends it back!",
        "Denied by {actor}!", "{actor} rejects {other_tok}!", "Big block from {actor}!",
        "{actor} with the emphatic block!",
    ),
    "steal": (
        "Stolen by {actor}!", "{actor} picks it off!", "{actor} with the steal.",
        "{actor} rips it away from {other_tok}!", "Steal, {actor}!", "{actor} takes it away!",
        "{actor} strips {other_tok}!", "Turnover forced by {actor}.", "{actor} pokes it free!",
        "{team} steals it, {actor_tok} with the takeaway.",
    ),
    "turnover": (
        "Turnover, {team}.", "{actor} turns it over.", "{team} gives it away.", "{actor} coughs it up.",
        "Giveaway by {actor}.", "{team} turns it over.", "{actor} gives it up.",
        "That's a turnover on {actor}.", "{actor} turns it over, {how}.", "Turnover on {team}, {how}.",
    ),
    "dreb": (
        "Rebound, {actor}.", "{actor} cleans it up.", "{actor} pulls it down.", "{team} rebounds.",
        "{actor} grabs it.", "{actor} secures the rebound.", "Board, {actor}.", "{actor} with the rebound.",
        "{team} secures it.",
    ),
    "oreb": (
        "Offensive rebound, {actor}!", "{actor} keeps it alive!", "Second chance for {team}!",
        "{actor} crashes the glass!", "{actor} with the offensive board.", "{team} gets it back!",
        "{actor} grabs the offensive rebound!", "{actor} hauls in the miss!",
        "{team} keeps the possession alive.",
    ),
    "team_reb": (
        "Team rebound, {team}.", "Rebound to {team}.", "{team} rebound.", "{team} gets it.",
        "Possession {team}.", "{team} takes over.", "{team} with the rebound.", "Rebound goes to {team}.",
    ),
    "assist": (
        "{actor} with the assist.", "Assist, {actor}.", "{actor} sets it up.", "Dished by {actor}.",
        "{actor} with the dime.", "Feed from {actor}.", "Courtesy of {actor}.", "{actor} finds {other_tok}.",
    ),
    "release3": (
        "{actor}, for three...", "{actor} from deep...", "{actor} for three...", "{actor} launches from deep...",
        "Three-point try, {actor}...", "{actor} lets it fly...", "{actor} from downtown...",
        "{actor} fires from deep...",
    ),
    "release": (
        "{actor} {pull_up}...", "{actor} rises...", "{actor} shoots...", "{actor} lets it go...",
        "{actor} puts it up...", "Shot by {actor}...", "{actor}, {zone}...", "{actor} with the shot...",
        "Up goes the shot...",
    ),
    "ft_attempt": (
        "{actor} at the line.", "{actor} to the line.", "Free throw, {actor}.", "{actor} steps up.",
        "{actor} shooting a free throw.", "At the line, {actor}.", "{actor} with the free throw.",
        "{actor} lines it up.",
    ),
    "possession": (
        "{team} ball.", "{team} takes over.", "Possession, {team}.", "{team} has it.", "{team} with the ball.",
        "Here comes {team}.", "{team} on the attack.", "Ball goes to {team}.", "{team} in possession.",
    ),
    "bring_up": (
        "{actor} brings it up.", "{actor_tok} walks it up for {team}.", "{actor} brings it across.",
        "{actor_tok} with the ball for {team}.", "{actor} sets up the offense.",
        "{team} brings it up, {actor_tok} handling.", "{actor_tok} initiates for {team}.",
        "{actor_tok} in control for {team}.", "{actor} brings it over half court.",
    ),
    "foul": (
        "Foul on {actor}.", "Whistle, {actor} is called for the foul.", "{actor} picks up a foul.",
        "That's a foul on {actor}.", "{actor} called for the foul.", "Whistle. Foul, {actor}.",
        "Foul called on {actor_tok} of {team}.", "{actor} is whistled for a foul.",
        "That's foul {pf_word} on {actor}.",
    ),
    "whistle": (
        "Whistle on the floor.", "Foul called.", "There's a whistle.", "Whistle, foul called.",
        "We have a foul.", "Foul on the play.", "Whistle blows.", "Play stopped, foul.",
    ),
    "timeout": (
        "Timeout, {team}.", "{team} calls timeout.", "{team} takes a timeout.",
        "Timeout on the floor, called by {team}.", "{team} stops the clock with a timeout.",
        "That's a timeout, {team}.", "{team} uses a timeout.", "Here's a timeout from {team}.",
        "Timeout on the floor.", "Stoppage on the floor.",
    ),
    "substitution": (
        "{actor_tok} checks in for {other_tok}.", "{actor_tok} in, {other_tok} out.",
        "Change for {team}: {actor_tok} replaces {other_tok}.", "{other_tok} heads to the bench, {actor_tok} in.",
        "{team} sub: {actor_tok} for {other_tok}.", "{actor_tok} comes on for {other_tok}.",
        "Substitution, {team}: {actor_tok} in.", "{actor_tok} enters for {team}.",
        "{other_tok} subs out for {actor_tok}.",
    ),
    "period_start": (
        "We're underway in the {period}.", "Here we go, {period} underway.", "The {period} is underway.",
        "And we're off in the {period}.", "Ball is live, {period} begins.", "Start of the {period}.",
        "The {period} gets going.", "Play begins in the {period}.", "And we're underway.",
    ),
    "period_end": (
        "That's the end of the {period}.", "The {period} is in the books.", "End of the {period}.",
        "The horn sounds on the {period}.", "That does it for the {period}.", "The {period} comes to a close.",
        "Buzzer, end of the {period}.", "And that wraps up the {period}.", "The {period} is over.",
        "End of the period.",
    ),
    "score_update": (
        "{leader} leads {lead_score}.", "Score check: {home} {h}, {away} {a}.", "It's {lead_score}, {leader}.",
        "{leader} up {margin_word}, {lead_score}.", "{home} {h}, {away} {a}.", "{leader} on top, {lead_score}.",
        "{leader} ahead by {margin_word}.", "The score: {lead_score}, {leader} in front.",
    ),
    "run": (
        "{run_team} on a {run_score} run.", "That's a {run_score} run for {run_team}.",
        "{run_team} has gone on a {run_score} run.", "{run_score} run, {run_team}.",
        "{run_team} rolling, {run_score} run.", "A {run_score} spurt for {run_team}.",
        "{run_team} with a {run_score} burst.", "Momentum {run_team}: a {run_score} run.",
    ),
    # ---- hedge families: attempt, zone and team only; never an outcome or a point value (§7A.2)
    "hedge_shot": (
        "Shot from the {zone}, {actor}.", "{actor} puts it up from the {zone}.",
        "{actor} with a look from the {zone}.", "Attempt from the {zone} by {actor}.", "{actor} lets it go.",
        "{actor} with the attempt.", "{team} shot from the {zone}.", "Shot goes up, {actor}.",
        "{actor} rises and lets it go.", "Up it goes from {actor}.",
    ),
    "hedge_ft": (
        "{actor} at the line.", "Free throw up from {actor}.", "{actor} with the free throw.",
        "{actor} shoots it from the line.", "Free throw attempt, {actor}.", "{actor} steps to the line.",
        "From the line, {actor}.", "{actor} lets the free throw go.",
    ),
    "hedge_loose": (
        "Loose ball!", "Ball's loose.", "Scramble for it.", "Ball is up for grabs.", "Loose ball near the {zone}.",
        "Scramble near the {zone}.", "Battle for the ball.", "Fight for the loose ball.", "Who has it? Scramble.",
    ),
    "hedge_stop": (  # an unconfirmed dead-ball turnover (out of bounds, violation): the ball is not loose
        "Play is stopped.", "Stoppage on the floor.", "Whistle, play stopped.", "The whistle blows.",
        "We have a stoppage.", "Stoppage in play.", "Play halted.", "Whistle on the floor.",
    ),
    "hedge_rebound": (
        "Rebound, {team}.", "{team} rebound.", "{team} gets it.", "Board to {team}.", "{team} with it.",
        "Rebound to {team}.", "{team} secures it.", "Ball to {team}.", "{team} has the rebound.",
    ),
}

# Short add-ons appended to a made call when they fit the line cap.
ADDONS: dict[str, tuple[str, ...]] = {
    "assist": (
        "Assist, {assister}.", "{assister} with the dime.", "Feed from {assister}.", "{assister} set it up.",
        "Dished by {assister}.", "Nice find, {assister}.", "Courtesy of {assister}.", "Thanks to {assister}.",
    ),
    "streak": ("{streak_ord} straight make.", "{streak_ord} in a row.", "That's the {streak_ord} straight.",
               "{streak_ord} straight for {actor}."),
    "tail_lead": (
        "{leader} by {margin_word}.", "{leader} up {margin_word}.", "{leader} leads {lead_score}.",
        "It's {lead_score}, {leader}.", "{leader} in front, {lead_score}.", "Score: {lead_score} {leader}.",
        "{leader} on top by {margin_word}.", "{lead_score} now, {leader} on top.",
    ),
    "tail_lead_change": (
        "{leader} takes the lead.", "{leader} goes ahead.", "{leader} takes the lead, {lead_score}.",
        "And {leader} goes ahead.", "Lead change, {leader} up {margin_word}.", "{leader} goes ahead {lead_score}.",
    ),
    "tail_tie": (
        "Tied at {h}.", "All tied at {h}.", "We're tied at {h}.", "Even at {h}.", "Tie game, {h}-{a}.",
        "All square at {h}.", "Level at {h}.", "Dead even at {h}.",
    ),
    # a tie at 0-0 ("Level at 0" read oddly): score updates say it is still scoreless, made calls add no tail
    "tail_scoreless": ("Still scoreless.", "No score yet.", "Scoreless so far.", "Nothing on the board yet."),
}

# --------------------------------------------------------------------------- coach cards (flat, numeric)

COACH: dict[str, tuple[str, ...]] = {
    "made": (
        "{actor}: {value_word} made. {pts} points, {fg} FG.", "Make, {actor}, {value_word}. {pts} points on {fg}.",
        "{team} score, {actor_tok}. Now {pts} points, {fg}.", "{actor} {value_word} made, {pts} points, {fg} shooting.",
        "{actor}: {value_word} made.",
    ),
    "miss": ("{actor} miss. {fg} FG.", "Miss, {actor}: {fg} from the field.", "{actor} missed, now {fg}.",
             "{team} miss, {actor_tok}."),
    "block": ("{actor} block on {other_tok}.", "Block, {actor}. {blk} blocks.", "{team} block by {actor_tok}, {blk} total.",
              "{actor}: {blk} blocks."),
    "steal": ("{actor} steal from {other_tok}.", "Steal, {actor}. {stl} steals.", "{opp} turnover, steal {actor}.",
              "{actor}: {stl} steals. {opp} {opp_tov} turnovers."),
    "turnover": ("{actor} turnover. {team} {team_tov} turnovers.", "Turnover, {actor}.", "{team} turnover, {team_tov} total.",
                 "{actor} turnover, {tov} on the game."),
    "rebound": ("{actor} {reb_type} rebound. {reb} rebounds.", "Rebound, {actor}: {reb} total.",
                "{team} {reb_type} rebound, {actor_tok}.", "{actor}: {reb} rebounds."),
    "team_rebound": ("{team} team rebound.", "Team rebound, {team}.", "{team} rebound, no individual credit.",
                     "Rebound: {team}, team."),
    "foul": ("Foul, {actor}: {pf} fouls.", "{actor} foul. {pf} personal fouls.", "{actor} now has {pf} fouls.",
             "Foul on {actor}. {team} {team_pf} team fouls.", "Foul on {actor}."),
    "hedge": ("{actor} attempt, {zone}. Result unconfirmed.", "{team} shot, result pending.",
              "Shot by {actor}, outcome unconfirmed.", "{team} attempt from {zone}, pending."),
    "hedge_rebound": ("Rebound {team}, unconfirmed.", "{team} rebound, pending.", "Rebound: {team}, unconfirmed.",
                      "{team} board, pending confirmation."),
    "hedge_loose": ("Loose ball, result pending.", "Possession unclear, pending.", "Loose ball, unconfirmed.",
                    "Ball loose, awaiting confirmation."),
    "hedge_stop": ("Stoppage, call pending.", "Play stopped, unconfirmed.", "Stoppage, awaiting confirmation.",
                   "Dead ball, pending."),
    "period_end": ("End of {period}. {home} {h}, {away} {a}.", "{period} over: {home} {h}, {away} {a}.",
                   "End {period}: {home} {h}, {away} {a}.", "{period} final. {home} {h}, {away} {a}.",
                   "End of {period}."),
    "run": ("{run_team} run: {run_score}.", "Run: {run_score} {run_team}.", "{run_team} on a {run_score} run.",
            "{run_score} run, {run_team}."),
}

# --------------------------------------------------------------------------- corrections (<= 12 words)

CORRECTION: dict[str, tuple[str, ...]] = {
    "no_basket": (
        "Correction: that basket does not count.", "Correction: no basket on that play.",
        "Correction: that shot did not go down.", "Correction: take those points off the board.",
        "Correction: the basket is waved off.", "Correction: that one did not count.",
        "Correction: scratch that basket.", "Correction: no good on that shot.",
    ),
    "was_good": (
        "Correction: that shot was good.", "Correction: that one counted.", "Correction: the basket counts.",
        "Correction: that shot went in.",
    ),
    "value": (
        "Correction: that's a {new_word}, not a {old_word}.", "Correction: make that a {new_word} for {new}.",
        "Correction: that shot counts for {new_word}.", "Correction: it's a {new_word}-pointer, not a {old_word}.",
        "Correction: {new_word} points on that basket, not {old_word}.", "Correction: that one is worth {new_word}.",
        "Correction: score that as a {new_word}.", "Correction: {new} gets {new_word} on that one, not {old_word}.",
    ),
    "scorer": (
        "Correction: that bucket goes to {new}, not {old}.", "Correction: {new} with that basket, not {old}.",
        "Correction: the basket belongs to {new}.", "Correction: credit {new} with that score, not {old}.",
        "Correction: that was {new}, not {old}, with the score.", "Correction: {new} scored that one.",
        "Correction: make that {new} with the basket.", "Correction: {new}, not {old}, gets the bucket.",
    ),
    "foul": (
        "Correction: the foul is on {new}, not {old}.", "Correction: that foul goes to {new}.",
        "Correction: {new} picks up that foul, not {old}.", "Correction: foul on {new}.",
    ),
    "timeout": ("Correction: that timeout belongs to {new_team}.", "Correction: timeout {new_team}.",
                "Correction: {new_team} called that timeout."),
    "credit": ("Correction: credit {new} on that play.", "Correction: that was {new}, not {old}.",
               "Correction: make that {new}.", "Correction: {new} on that play, not {old}."),
    "generic": (
        "Correction on that last call.", "Correction: scratch that last call.", "Correction: disregard that last call.",
        "Correction: that call was wrong.", "Correction: let's fix that last call.", "Correction: that last call is changed.",
    ),
}

_SCORE_LINE = "{home} {h}, {away} {a}."
_TURNOVER_HOW = {"travel": "traveling", "out_of_bounds": "out of bounds", "shot_clock": "shot-clock violation",
                 "bad_pass": "bad pass", "offensive_foul": "offensive foul"}
_BRING_UP_CAUSES = frozenset({"made_shot", "inbound", "free_throw", "period_start", "defensive_rebound"})
_MADE_FAMILIES = frozenset({"made2", "made3", "layup", "dunk", "ft_made"})
# AD: families after which the score is read (one possession ended).
_AD_SCORE_AFTER = frozenset({"made2", "made3", "layup", "dunk", "ft_made", "dreb", "team_reb", "steal", "turnover",
                             "period_end", "score_update"})


@dataclass
class TemplateConfig:
    """Word caps per family (narration.md §9.2 at ~3.5 words/s; design proposals, unverified) and
    composition rules."""

    max_words: dict[str, int] = field(default_factory=lambda: {
        "release": 4, "release3": 4, "ft_attempt": 5, "made2": 8, "made3": 8, "layup": 8, "dunk": 10,
        "ft_made": 5, "ft_miss": 5, "miss": 5, "block": 10, "steal": 8, "turnover": 8, "oreb": 6, "dreb": 4,
        "team_reb": 4, "assist": 6, "possession": 6, "bring_up": 6, "foul": 10, "whistle": 5, "timeout": 10,
        "substitution": 10, "period_start": 12, "period_end": 12, "score_update": 12, "run": 12,
        "hedge_shot": 8, "hedge_ft": 6, "hedge_loose": 6, "hedge_stop": 5, "hedge_rebound": 4,
    })
    line_cap: dict[str, int] = field(default_factory=lambda: {
        "play_by_play": 12, "color": 35, "coach": 12, "accessibility": 6, "correction": 12})  # match ValidatorConfig
    assist_extra_words: int = 4  # assist add-on may extend a made call by this much (design proposal)
    streak_extra_words: int = 4  # streak add-on allowance (design proposal)
    score_tail_max_margin: int = 6  # score tail on makes within two possessions, ties, lead changes; tuned
    ad_names: bool = True  # AD shot calls carry the shooter token (identity tiers apply when rendered)
    lru_window: int = 3  # none of the last 3 variants of a family repeats (narration.md §3.4); sourced
    repetition_jaccard: float = 0.8  # never pick a variant the validator would reject as a repeat (spec)


@dataclass
class RenderedLine:
    text: str  # token form; "" when there is nothing to say on this channel
    family: str  # e.g. "made3", "hedge_shot", "ad", "correction:scorer"
    variant: int = -1  # index in the family's variant tuple (-1 for fixed lines)
    cap: int = 0  # word cap the line was built to

    @property
    def hedged(self) -> bool:
        """Hedge wording (counts toward the director's 1-in-3 hedge cap)."""
        return self.family.startswith(("hedge_", "coach:hedge"))


def family_cap(family: str, config: TemplateConfig | None = None) -> int:
    """Word cap for a play-by-play family (12 when unknown)."""
    return (config or TemplateConfig()).max_words.get(family, 12)


# --------------------------------------------------------------------------- public API


def render(events: list[Event], facts: dict, channel: str, rng: random.Random, hedge: bool = False) -> str:
    """One deterministic line (``{Pnn}`` tokens) for ``events`` on ``channel``; "" if nothing to say.

    ``hedge`` is forced on when the packet is hedge-only. Hedge lines never state an outcome or a
    point value.
    """
    return render_line(events, facts, channel, rng, hedge).text


def render_line(events: list[Event], facts: Mapping[str, Any], channel: str, rng: random.Random,
                hedge: bool = False, *, config: TemplateConfig | None = None,
                history: dict[str, deque[int]] | None = None) -> RenderedLine:
    """Like :func:`render` but returns the family, variant and cap (for logging and LRU)."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}; expected one of {CHANNELS}")
    cfg = config or TemplateConfig()
    evs = _event_facts(events, facts)
    if channel == "correction":
        return _render_correction(facts, rng, cfg, history)
    if channel == "color":
        return _render_color(facts, rng, cfg, history)
    hedged = bool(hedge or facts.get("hedge_only"))
    family, ev, extra = _plan(evs, facts, hedged)
    if not family:
        return RenderedLine("", "")
    ctx = _context(ev, facts, extra)
    if channel == "accessibility":
        return _render_ad(family, ctx, facts, cfg)
    if channel == "coach":
        return _render_coach(family, ctx, facts, rng, cfg, history)
    return _render_pbp(family, ctx, facts, rng, cfg, history)


class TemplateRenderer:
    """Stateful renderer: seeded rng plus per-family LRU so no variant repeats within 3 uses."""

    def __init__(self, config: TemplateConfig | None = None, seed: int = 0) -> None:
        self.config = config or TemplateConfig()
        self.rng = random.Random(seed)
        self._history: dict[str, deque[int]] = {}

    def render(self, events: list[Event], facts: Mapping[str, Any], channel: str, hedge: bool = False) -> str:
        return self.render_line(events, facts, channel, hedge).text

    def render_line(self, events: list[Event], facts: Mapping[str, Any], channel: str,
                    hedge: bool = False) -> RenderedLine:
        return render_line(events, facts, channel, self.rng, hedge, config=self.config, history=self._history)


# --------------------------------------------------------------------------- planning


def _event_facts(events: Sequence[Event], facts: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_id = {f.get("event_id"): f for f in facts.get("events") or () if isinstance(f, Mapping)}
    out = []
    for e in events:
        f = by_id.get(e.event_id)
        if f is None:  # not in the packet: team-level only, nothing outcome-dependent
            f = {"event_id": e.event_id, "kind": e.kind.value, "outcome_ok": False,
                 "retracted": e.status.value == "retracted"}
            log.debug("event %s missing from facts; rendering team-level hedge", e.event_id)
        out.append(dict(f))
    return out


def _first(evs: list[dict[str, Any]], kind: str) -> dict[str, Any] | None:
    return next((f for f in evs if f.get("kind") == kind and not f.get("retracted")), None)


def _is_ft(f: Mapping[str, Any]) -> bool:
    return f.get("shot_type") == "free_throw" or f.get("value") == 1


def _plan(evs: list[dict[str, Any]], facts: Mapping[str, Any],
          hedged: bool) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """(family, primary event fact, extras) for the play-by-play bank."""
    extra: dict[str, Any] = {}

    def hedge_of(f: Mapping[str, Any]) -> bool:
        return hedged or not f.get("outcome_ok", False)

    made = _first(evs, "shot_made")
    if made is not None:
        if hedge_of(made):
            return ("hedge_ft" if _is_ft(made) else "hedge_shot"), made, extra
        assist = _first(evs, "assist")
        if assist is not None and assist.get("outcome_ok") and assist.get("actor"):
            extra["assister"] = "{" + assist["actor"] + "}"
        shot_type = made.get("shot_type")
        if _is_ft(made):
            fam = "ft_made"
        elif shot_type == "dunk":
            fam = "dunk"
        elif shot_type == "layup":
            fam = "layup"
        else:
            fam = "made3" if made.get("value") == 3 else "made2"
        return fam, made, extra
    block = _first(evs, "block")
    if block is not None:
        if hedge_of(block):
            return "hedge_shot", _shooter_of_block(block, facts), extra
        return "block", block, extra
    for kind in ("steal", "turnover"):
        f = _first(evs, kind)
        if f is not None:
            if not hedge_of(f):
                return kind, f, extra
            live_ball = kind == "steal" or f.get("turnover_type") in (None, "steal", "bad_pass")
            return ("hedge_loose" if live_ball else "hedge_stop"), f, extra
    reb = _first(evs, "rebound")
    if reb is not None:
        if hedge_of(reb):
            return "hedge_rebound", reb, extra
        if not reb.get("actor"):
            return "team_reb", reb, extra
        return ("oreb" if reb.get("offensive") else "dreb"), reb, extra
    miss = _first(evs, "shot_missed")
    if miss is not None:
        if hedge_of(miss):
            return ("hedge_ft" if _is_ft(miss) else "hedge_shot"), miss, extra
        return ("ft_miss" if _is_ft(miss) else "miss"), miss, extra
    foul = _first(evs, "foul")
    if foul is not None:
        return ("whistle" if hedge_of(foul) or not foul.get("actor") else "foul"), foul, extra
    attempt = _first(evs, "shot_attempt")
    if attempt is not None:
        if _is_ft(attempt):
            return "ft_attempt", attempt, extra
        return ("release3" if attempt.get("value") == 3 and not hedge_of(attempt) else "release"), attempt, extra
    assist = _first(evs, "assist")
    if assist is not None:
        return ("" if hedge_of(assist) else "assist"), assist, extra
    for kind in ("timeout", "substitution", "period_end", "period_start"):
        f = _first(evs, kind)
        if f is not None:
            return kind, f, extra
    poss = _first(evs, "possession")
    if poss is not None:
        fam = "bring_up" if poss.get("cause") in _BRING_UP_CAUSES and poss.get("actor") else "possession"
        return fam, poss, extra
    if not evs and (facts.get("score") or {}).get("confirmed"):
        return "score_update", {}, extra
    return "", {}, extra


def _shooter_of_block(block: Mapping[str, Any], facts: Mapping[str, Any]) -> dict[str, Any]:
    """A hedged block is narrated as the shooter's attempt (shooter = ``other``, opposing team)."""
    shot = {k: v for k, v in block.items() if k not in ("actor", "other", "team", "team_side")}
    if block.get("other"):
        shot["actor"] = block["other"]
    side = block.get("team_side")
    if side in ("home", "away"):
        opp = "away" if side == "home" else "home"
        name = ((facts.get("teams") or {}).get(opp) or {}).get("name")
        shot["team_side"] = opp
        if name:
            shot["team"] = name
    return shot


def _context(ev: Mapping[str, Any], facts: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, str]:
    """Placeholder values; missing facts are simply absent (variants needing them are skipped)."""
    c: dict[str, str] = {}
    teams = facts.get("teams") or {}
    side = ev.get("team_side")
    if ev.get("team"):
        c["team"] = str(ev["team"])
    if side in ("home", "away"):
        opp = teams.get("away" if side == "home" else "home") or {}
        if opp.get("name"):
            c["opp"] = str(opp["name"])
        _team_counts(c, teams.get(side) or {}, opp)
    if ev.get("actor"):
        c["actor_tok"] = "{" + str(ev["actor"]) + "}"
    if "actor_tok" in c or "team" in c:
        c["actor"] = c.get("actor_tok") or c["team"]
    if ev.get("other"):
        c["other_tok"] = "{" + str(ev["other"]) + "}"
    if ev.get("zone"):
        c["zone"] = str(ev["zone"])
    if ev.get("value"):
        c["value_word"] = int_to_words(int(ev["value"]))
    if ev.get("shot_type") == "jumper":  # shot-type wording only when the engine says it was a jump shot
        c.update(jumper="jumper", pull_up="pulls up")
    if ev.get("kind") == "rebound":
        c["reb_type"] = "offensive" if ev.get("offensive") else "defensive"
    how = _TURNOVER_HOW.get(str(ev.get("turnover_type")))
    if how:
        c["how"] = how
    _player_counts(c, (facts.get("players") or {}).get(ev.get("actor") or "") or {})
    _score_context(c, facts)
    _streak_context(c, facts, ev.get("actor"))
    clock = facts.get("clock") or {}
    if clock.get("period_text"):
        c["period"] = str(clock["period_text"])
    run = facts.get("run") or {}
    if run.get("text") and run.get("team"):
        c["run_team"] = str(run["team"])
        c["run_score"] = f"{run['points']}-{run['opp_points']}"
    c.update({k: str(v) for k, v in extra.items()})
    return c


def _player_counts(c: dict[str, str], p: Mapping[str, Any]) -> None:
    for k in ("pts", "fg", "reb", "blk", "stl", "tov", "pf"):
        if p.get(k) is not None:
            c[k] = str(p[k])
    if p.get("pf"):
        c["pf_word"] = int_to_words(int(p["pf"]))


def _team_counts(c: dict[str, str], mine: Mapping[str, Any], opp: Mapping[str, Any]) -> None:
    if mine.get("tov") is not None:
        c["team_tov"] = str(mine["tov"])
    if mine.get("pf") is not None:
        c["team_pf"] = str(mine["pf"])
    if opp.get("tov") is not None:
        c["opp_tov"] = str(opp["tov"])


def _score_context(c: dict[str, str], facts: Mapping[str, Any]) -> None:
    score = facts.get("score") or {}
    teams = facts.get("teams") or {}
    if not score.get("confirmed") or score.get("home") is None:
        return  # templates speak the score only from confirmed values
    h, a = int(score["home"]), int(score["away"])
    c.update(home=str((teams.get("home") or {}).get("name", "Home")),
             away=str((teams.get("away") or {}).get("name", "Away")), h=str(h), a=str(a))
    if h != a:
        c.update(leader=str(score.get("leader")), margin_word=int_to_words(abs(h - a)),
                 lead_score=f"{max(h, a)}-{min(h, a)}")


def _streak_context(c: dict[str, str], facts: Mapping[str, Any], actor: str | None) -> None:
    if not actor:
        return
    rx = re.compile(r"^([a-z-]+) straight make for \{" + re.escape(actor) + r"\}$")
    for phrase in facts.get("derived") or ():
        m = rx.match(str(phrase))
        if m:
            c["streak_ord"] = m.group(1)
            return


# --------------------------------------------------------------------------- choosing variants


def _cap_sentences(text: str) -> str:
    """Upper-case the first letter of each sentence (placeholders can start lower-case)."""
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)


# a count of one takes the singular ("now has 1 fouls"); the counted nouns the templates use
_ONE_PLURAL_RE = re.compile(
    r"\b(1|one) ((?:personal |team |offensive |defensive )?(?:foul|point|rebound|steal|block|turnover|assist))s\b",
    re.IGNORECASE)


def _fill(variants: Sequence[str], ctx: Mapping[str, str]) -> list[tuple[int, str]]:
    out = []
    for i, v in enumerate(variants):
        try:
            out.append((i, _cap_sentences(_ONE_PLURAL_RE.sub(r"\1 \2", v.format_map(ctx)))))
        except (KeyError, IndexError, ValueError):
            continue
    return out


def _ordered(variants: Sequence[str], ctx: Mapping[str, str], cap: int, rng: random.Random,
             key: str, history: dict[str, deque[int]] | None) -> list[tuple[int, str]]:
    """Fillable variants within ``cap`` in rng order, least-recently-used first."""
    filled = _fill(variants, ctx)
    if not filled:
        return []
    fitting = [p for p in filled if count_words(p[1]) <= cap]
    if not fitting:
        fitting = [min(filled, key=lambda p: count_words(p[1]))]
        log.debug("no %s variant fits %d words; using the shortest", key, cap)
    rng.shuffle(fitting)
    if history is not None and key in history:
        used = set(history[key])
        fitting.sort(key=lambda p: p[0] in used)  # stable: unused variants keep rng order, first
    return fitting


def _not_repeat(text: str, recent: Sequence[str], thr: float) -> bool:
    return all(jaccard(text, r) < thr for r in recent)


def _pick(variants: Sequence[str], ctx: Mapping[str, str], cap: int, rng: random.Random, key: str,
          history: dict[str, deque[int]] | None, recent: Sequence[str], cfg: TemplateConfig,
          compose: Any = None) -> tuple[int, str] | None:
    options = _ordered(variants, ctx, cap, rng, key, history)
    if not options:
        return None
    best: tuple[float, int, str] | None = None
    for i, base in options:
        text = compose(base) if compose else base
        if _not_repeat(text, recent, cfg.repetition_jaccard):
            _remember(history, key, i, cfg)
            return i, text
        sim = max((jaccard(text, r) for r in recent), default=0.0)
        if best is None or sim < best[0]:
            best = (sim, i, text)
    assert best is not None
    _remember(history, key, best[1], cfg)
    return best[1], best[2]


def _remember(history: dict[str, deque[int]] | None, key: str, idx: int, cfg: TemplateConfig) -> None:
    if history is not None:
        history.setdefault(key, deque(maxlen=cfg.lru_window)).append(idx)


def _recent(facts: Mapping[str, Any]) -> list[str]:
    return [str(s) for s in facts.get("recent_lines") or ()]


# --------------------------------------------------------------------------- channel renderers


def _render_pbp(family: str, ctx: dict[str, str], facts: Mapping[str, Any], rng: random.Random,
                cfg: TemplateConfig, history: dict[str, deque[int]] | None) -> RenderedLine:
    cap = family_cap(family, cfg)
    line_cap = cfg.line_cap["play_by_play"]
    if family == "score_update" and "leader" not in ctx:
        variants = _tie_variants(ctx)
    else:
        variants = PBP[family]
    recent = _recent(facts)

    def compose(base: str) -> str:
        if family not in _MADE_FAMILIES:
            return base
        return _with_addons(base, family, ctx, facts, rng, cfg, cap, line_cap)

    picked = _pick(variants, ctx, cap, rng, family, history, recent, cfg, compose)
    if picked is None:
        return RenderedLine("", family)
    idx, text = picked
    return RenderedLine(text, family, idx, min(line_cap, cap + cfg.assist_extra_words + cfg.streak_extra_words)
                        if family in _MADE_FAMILIES else cap)


def _with_addons(base: str, family: str, ctx: Mapping[str, str], facts: Mapping[str, Any], rng: random.Random,
                 cfg: TemplateConfig, cap: int, line_cap: int) -> str:
    text = base
    tail_key = _tail_key(ctx, facts, cfg)
    if tail_key:
        text = _append(text, ADDONS[tail_key], ctx, rng, cap)
    if "assister" in ctx:
        text = _append(text, ADDONS["assist"], ctx, rng, min(line_cap, cap + cfg.assist_extra_words))
    if "streak_ord" in ctx:
        text = _append(text, ADDONS["streak"], ctx, rng,
                       min(line_cap, cap + cfg.assist_extra_words + cfg.streak_extra_words))
    return text


def _tail_key(ctx: Mapping[str, str], facts: Mapping[str, Any], cfg: TemplateConfig) -> str | None:
    if "h" not in ctx:
        return None
    if "leader" not in ctx:
        return "tail_tie" if ctx.get("h") != "0" else None  # 0-0 after a make: the board has not caught up
    derived = " | ".join(str(d) for d in facts.get("derived") or ())
    if f"{ctx['leader']} takes the lead" in derived:
        return "tail_lead_change"
    score = facts.get("score") or {}
    return "tail_lead" if int(score.get("margin", 99)) <= cfg.score_tail_max_margin else None


def _tie_variants(ctx: Mapping[str, str]) -> tuple[str, ...]:
    return ADDONS["tail_scoreless"] if ctx.get("h") == "0" else ADDONS["tail_tie"]


def _append(text: str, variants: Sequence[str], ctx: Mapping[str, str], rng: random.Random, cap: int) -> str:
    room = cap - count_words(text)
    fits = [s for _, s in _fill(variants, ctx) if count_words(s) <= room]
    return f"{text} {rng.choice(fits)}" if fits else text


def _render_coach(family: str, ctx: dict[str, str], facts: Mapping[str, Any], rng: random.Random,
                  cfg: TemplateConfig, history: dict[str, deque[int]] | None) -> RenderedLine:
    key = _coach_key(family)
    if family == "score_update" and "run_team" in ctx:
        key = "run"
    if not key:
        return RenderedLine("", "coach")
    picked = _pick(COACH[key], ctx, cfg.line_cap["coach"], rng, f"coach:{key}", history, _recent(facts), cfg)
    if picked is None:
        return RenderedLine("", f"coach:{key}")
    return RenderedLine(picked[1], f"coach:{key}", picked[0], cfg.line_cap["coach"])


def _coach_key(family: str) -> str:
    if family in _MADE_FAMILIES:
        return "made"
    return {"miss": "miss", "ft_miss": "miss", "block": "block", "steal": "steal", "turnover": "turnover",
            "dreb": "rebound", "oreb": "rebound", "team_reb": "team_rebound", "foul": "foul",
            "hedge_shot": "hedge", "hedge_ft": "hedge", "hedge_rebound": "hedge_rebound",
            "hedge_loose": "hedge_loose", "hedge_stop": "hedge_stop", "period_end": "period_end"}.get(family, "")


def _render_ad(family: str, ctx: dict[str, str], facts: Mapping[str, Any], cfg: TemplateConfig) -> RenderedLine:
    line = _ad_core(family, ctx, cfg)
    if family in _AD_SCORE_AFTER and "h" in ctx:
        score = _SCORE_LINE.format_map(ctx)
        line = f"{line} {score}" if line else score
    return RenderedLine(_cap_sentences(line), "ad", -1, cfg.line_cap["accessibility"])


def _ad_core(family: str, ctx: Mapping[str, str], cfg: TemplateConfig) -> str:
    """Fixed AD lexicon (RESEARCH.md §7A.8): one short call per possession event."""
    who = ctx.get("actor_tok") if cfg.ad_names else None
    result = {"made2": "good", "made3": "good", "layup": "good", "dunk": "good", "ft_made": "good",
              "miss": "no good", "ft_miss": "no good", "block": "blocked"}.get(family)
    if family in ("ft_made", "ft_miss", "hedge_ft", "ft_attempt"):
        core = f"{who} free throw" if who else "free throw"
    elif family in _MADE_FAMILIES or family in ("miss", "hedge_shot", "release", "release3"):
        core = f"{who} shot" if who else "shot"
        core += f", {ctx['zone']}" if ctx.get("zone") else ""
    elif family == "block":
        core = f"shot, {ctx['zone']}" if ctx.get("zone") else "shot"
    else:
        return _ad_other(family, ctx)
    return core + (f", {result}" if result else "") + "."


def _ad_other(family: str, ctx: Mapping[str, str]) -> str:
    team, opp, period = ctx.get("team"), ctx.get("opp"), ctx.get("period")
    if family in ("dreb", "oreb", "team_reb", "hedge_rebound"):
        return f"Rebound, {team}." if team else "Rebound."
    if family == "steal":
        return f"Turnover, {opp}." if opp else "Turnover."
    if family == "turnover":
        return f"Turnover, {team}." if team else "Turnover."
    if family == "hedge_loose":
        return "Loose ball."
    if family == "hedge_stop":
        return "Stoppage."
    if family in ("possession", "bring_up"):
        return f"{team} ball." if team else ""
    if family == "period_start":
        return f"Start of {period}." if period else "Start of period."
    if family == "period_end":
        return f"End of {period}." if period else "End of period."
    if family in ("foul", "whistle"):
        return f"Foul, {team}." if team and family == "foul" else "Whistle."
    if family == "timeout":
        return f"Timeout, {team}." if team else "Timeout."
    return ""


def _render_correction(facts: Mapping[str, Any], rng: random.Random, cfg: TemplateConfig,
                       history: dict[str, deque[int]] | None) -> RenderedLine:
    corr = facts.get("correction") or {}
    key = _correction_key(corr)
    ctx: dict[str, str] = {}
    if corr.get("new_actor"):
        ctx["new"] = "{" + str(corr["new_actor"]) + "}"
    if corr.get("old_actor"):
        ctx["old"] = "{" + str(corr["old_actor"]) + "}"
    if corr.get("new_value"):
        ctx["new_word"] = int_to_words(int(corr["new_value"]))
    if corr.get("old_value"):
        ctx["old_word"] = int_to_words(int(corr["old_value"]))
    if corr.get("new_team"):
        ctx["new_team"] = str(corr["new_team"])
    _score_context(ctx, facts)
    cap = cfg.line_cap["correction"]

    def compose(base: str) -> str:
        if key in ("no_basket", "was_good", "value", "scorer") and "h" in ctx:
            tail = _SCORE_LINE.format_map(ctx)
            if count_words(base) + count_words(tail) <= cap:
                return f"{base} {tail}"
        return base

    picked = _pick(CORRECTION[key], ctx, cap, rng, f"correction:{key}", history, _recent(facts), cfg, compose)
    if picked is None:
        picked = _pick(CORRECTION["generic"], ctx, cap, rng, "correction:generic", history, _recent(facts), cfg)
        key = "generic"
    assert picked is not None
    return RenderedLine(picked[1], f"correction:{key}", picked[0], cap)


def _correction_key(c: Mapping[str, Any]) -> str:
    kind, old_kind = c.get("kind"), c.get("old_kind", c.get("kind"))
    # The engine revises a shot result in place (SHOT_MISSED <-> SHOT_MADE, same event_id) and
    # retracts a shot that never happened. A retracted miss is NOT a make: no "was good" line then.
    if old_kind == "shot_made" and (c.get("retracted") or kind == "shot_missed"):
        return "no_basket"
    if old_kind == "shot_missed" and kind == "shot_made" and not c.get("retracted"):
        return "was_good"
    old_v, new_v = c.get("old_value"), c.get("new_value")
    if kind in ("shot_made", "shot_attempt") and old_v and new_v and old_v != new_v:
        return "value"
    old_a, new_a = c.get("old_actor"), c.get("new_actor")
    if old_a and new_a and old_a != new_a:
        return {"shot_made": "scorer", "foul": "foul"}.get(str(kind), "credit")
    if kind == "timeout" and c.get("new_team") and c.get("new_team") != c.get("old_team"):
        return "timeout"
    return "generic"


def _render_color(facts: Mapping[str, Any], rng: random.Random, cfg: TemplateConfig,
                  history: dict[str, deque[int]] | None) -> RenderedLine:
    """Deterministic colour: one code-made ``derived`` phrase, else the run, else the score."""
    run_text = (facts.get("run") or {}).get("text")
    phrases = [str(p) for p in facts.get("derived") or () if str(p) != run_text]
    if facts.get("hedge_only"):  # a trusted event's phrase may still carry outcome words ("reaches 10 points")
        phrases = [p for p in phrases if not OUTCOME_RE.search(TOKEN_RE.sub(" ", p))]
    options = [_cap_sentences(p[0].upper() + p[1:] + ".") for p in phrases if p]
    recent = _recent(facts)
    cap = cfg.line_cap["color"]
    rng.shuffle(options)
    for text in options:
        if count_words(text) <= cap and _not_repeat(text, recent, cfg.repetition_jaccard):
            return RenderedLine(text, "color:derived", -1, cap)
    ctx = _context({}, facts, {})
    for family in ("run", "score_update"):
        variants = PBP[family] if family == "run" or "leader" in ctx else _tie_variants(ctx)
        picked = _pick(variants, ctx, cap, rng, f"color:{family}", history, recent, cfg)
        if picked is not None and _not_repeat(picked[1], recent, cfg.repetition_jaccard):
            return RenderedLine(picked[1], f"color:{family}", picked[0], cap)
    return RenderedLine("", "color")


def sentences(text: str) -> list[str]:
    """Sentence split used by callers that stream template text sentence by sentence."""
    return split_sentences(text)
