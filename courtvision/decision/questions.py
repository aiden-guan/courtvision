"""Every Jev question CourtVision sends (verbatim from jev_integration_design.md §5 and §11.1).

Static questions are built once at import. Dynamic option sets (line kinds, coach cards, guard
sentences) come from small builder functions. Never put raw numbers in instructions or criteria:
the only digits allowed are inside backticked state paths such as `storylines[0]` or `lines.s1`.
Question names are mapping keys and are not sent to the model, so each instruction is complete.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from typesafe_sdk import Box, Choice, Noul, NoulCriteria, Score


def _reversed(criteria: dict) -> dict:
    """Same options, opposite order: used for in-request order-ensembling (first-option bias)."""
    return dict(reversed(list(criteria.items())))


# ───────────────────────────── B1 possession_end ─────────────────────────────
PLAY_TYPE_INSTR = ("Which description best fits the action that ended the basketball possession "
                   "described in `possession`?")
PLAY_TYPE_CRITERIA: dict[str, str] = {   # Synergy-style ids; descriptions are working definitions (unverified vs Synergy)
    "Transition": "The offense attacked right after gaining the ball, before the defense was set",
    "Isolation": "The player who ended the possession went one-on-one against a defender, with no ball screen or handoff in that action",
    "PRBallHandler": "The possession ended with the ball handler who had just used a ball screen: a shot, a drawn shooting foul, or a turnover",
    "PRRollman": "The possession ended with the player who set the ball screen, after rolling, popping or slipping toward the basket",
    "Postup": "The possession ended with a player who received the ball with their back to the basket near the paint and played from there",
    "Spotup": "The possession ended with a player who caught the ball while standing and shot, or drove at a defender running out at them",
    "Handoff": "The possession ended with the player who took a handoff from a teammate",
    "Cut": "The possession ended with a player who moved toward the basket without the ball and received it near the rim",
    "OffScreen": "The possession ended with a player who ran off a screen set away from the ball and then received it",
    "OffRebound": "The possession ended with a shot right after an offensive rebound, before the offense reset",
    "Misc": "Something else ended the possession, such as a broken play, a violation, or a heave at the end of the clock",
    "unclear": "`possession.actions` is too incomplete to tell which action ended the possession",
}
PLAY_TYPE_LABELS: tuple[str, ...] = tuple(PLAY_TYPE_CRITERIA)

STAKES_LEVELS: tuple[str, ...] = ("routine", "notable", "high_stakes", "decisive")  # code names of the 4 levels

B1_POSSESSION_END = {
    "play_type": Choice(instructions=PLAY_TYPE_INSTR, criteria=PLAY_TYPE_CRITERIA),
    "play_type_rev": Choice(instructions=PLAY_TYPE_INSTR, criteria=_reversed(PLAY_TYPE_CRITERIA)),
    "summary_is_clear": Noul(
        instructions="Does `possession.actions` say clearly what the last player with the ball did to end the possession?",
        criteria=NoulCriteria(
            true="It names the final action, such as the shot, the pass right before it, or the turnover",
            false="The final action is missing, vague, or contradicted by `possession.end`",
        ),
    ),
    "moment_stakes": Score(
        instructions="How much was at stake in the moment described by `situation` and `last_play`?",
        criteria=[
            "Routine: an ordinary moment, early in a period or in a game that is not close",
            "Notable: a scoring run, a tie or lead change, a milestone within reach, or foul trouble for a key player",
            "High stakes: late in a close game, or a play that swings momentum in a close game",
            "Decisive: the final possessions of a one-possession game, or a play that likely decides the result",
        ],
    ),
    "highlight_play": Noul(
        instructions="Would a typical fan call the play in `last_play.description` a highlight?",
        criteria=NoulCriteria(
            true="Spectacular or game-changing: a dunk in traffic, a buzzer-beater, a chase-down block, a deep three that takes the lead late",
            false="A routine basket, free throw, rebound or ordinary possession",
        ),
    ),
}


# ───────────────────────────── B2 dead_ball_fan ─────────────────────────────
LINE_KINDS: dict[str, str] = {
    "stat_note": "A short note about a player's or team's numbers tonight, taken from `material.stat_note`",
    "tendency": "A pattern in how a team has played tonight, taken from `material.tendency`",
    "storyline": "A storyline from `storylines`, such as a milestone within reach",
    "silence": "Say nothing at this stoppage, because nothing in `material` or `storylines` is worth saying now",
}
MAX_STORYLINES = 3


def b2_dead_ball_fan(available_kinds: set[str], n_storylines: int) -> dict:
    """B2 questions: `next_line_kind` over the available kinds (silence always offered) plus one
    `storyline_due_i` Noul per flagged storyline (at most three)."""
    criteria = {k: v for k, v in LINE_KINDS.items() if k == "silence" or k in available_kinds}
    qs: dict[str, Any] = {
        "next_line_kind": Choice(
            instructions=("During the stoppage described in `situation`, which kind of line should a "
                          "live basketball commentator say next?"),
            criteria=criteria,
        )
    }
    for i in range(min(n_storylines, MAX_STORYLINES)):           # one Noul per item; never "how many"
        qs[f"storyline_due_{i}"] = Noul(
            instructions=(f"Does the storyline in `storylines[{i}]` connect to what just happened in "
                          "`last_play` or to the game situation in `situation`?"),
            criteria=NoulCriteria(
                true="Mentioning it now would follow naturally from the current moment",
                false="It is unrelated to the current moment and would sound random now",
            ),
        )
    return qs


# ───────────────────────────── B3 dead_ball_coach ─────────────────────────────
CARD_KEYS = ("card_a", "card_b", "card_c", "card_d", "card_e", "card_f", "card_g", "card_h")


def b3_dead_ball_coach(cards: Mapping[str, str]) -> dict:
    """cards: neutral letter key (CARD_KEYS, in shuffled order) -> code-rendered, pseudonymised card
    text (<= 12 words). 1..8 cards. Keys carry no digits, names or content, so the model sees only
    "the card stored at `cards.card_c`"; the text stays in the state (gap pass 2026-10-20)."""
    if not cards or not set(cards) <= set(CARD_KEYS):
        raise ValueError(f"card keys must be a non-empty subset of {CARD_KEYS}")
    qs: dict[str, Any] = {
        f"relevant__{key}": Noul(
            instructions=(f"Would the coaching staff of `situation.staff_team` want to see the card in "
                          f"`cards.{key}` during the stoppage described in `situation`?"),
            criteria=NoulCriteria(
                true="It bears on a decision the staff can make now, such as a substitution, a defensive coverage, a matchup or a timeout",
                false="It is background, out of date, or based on too few possessions to act on",
            ),
        )
        for key in cards
    }
    top = {key: f"The card stored at `cards.{key}`" for key in cards}   # never the card text itself
    top["none"] = "No card is important enough to interrupt the staff at this stoppage"
    qs["top_card"] = Choice(
        instructions=("Which single card, if any, should be pushed to the head coach's screen during the "
                      "stoppage described in `situation`?"),
        criteria=top,
    )
    return qs


# ───────────────────────────── B4 line_guard ─────────────────────────────
MAX_GUARD_SENTENCES = 2


def b4_line_guard(n_sentences: int) -> dict:
    """B4 questions for sentences `lines.s1`..`lines.s{n}` (n <= 2 per request)."""
    if not 1 <= n_sentences <= MAX_GUARD_SENTENCES:
        raise ValueError(f"B4 checks 1..{MAX_GUARD_SENTENCES} sentences per request, got {n_sentences}")
    qs: dict[str, Any] = {}
    for i in range(1, n_sentences + 1):
        s = f"`lines.s{i}`"
        qs[f"supported_s{i}"] = Noul(
            instructions=(f"Numbers in {s} were already checked by code, so ignore them. Is every name, "
                          f"action and description in {s} stated in, or directly implied by, `facts`?"),
            criteria=NoulCriteria(
                true="Every person, team, action and description in the line appears in `facts` or follows directly from them",
                false="The line says something that `facts` do not state or imply, or it contradicts `facts`",
            ),
        )
        qs[f"unlisted_s{i}"] = Noul(
            instructions=f"Does {s} mention a player, team, play or event that does not appear anywhere in `facts`?",
            criteria=NoulCriteria(
                true="It mentions at least one player, team, play or event that is not in `facts`",
                false="Every player, team, play and event it mentions appears in `facts`",
            ),
        )
        qs[f"speculation_s{i}"] = Noul(
            instructions=f"Does {s} state a guess, prediction or opinion as if it were an established fact?",
            criteria=NoulCriteria(
                true="It presents a guess, prediction or opinion as fact, such as what will happen next or why a player is struggling",
                false="It only reports what happened, or it clearly marks an opinion as an opinion",
            ),
        )
    return qs


# ───────────────────────────── B5 event_ambiguity ─────────────────────────────
SHOT_OUTCOME_INSTR = "Based only on the camera cues in `shot`, did the shot go down through the hoop?"
SHOT_OUTCOME_CRITERIA = {
    "make": "The cues show the ball went down through the rim",
    "miss": "The cues show the ball did not go down through the rim",
    "unclear": "The cues are missing or conflict, so the outcome cannot be told from them",
}
B5_SHOT_OUTCOME = {
    "shot_outcome": Choice(instructions=SHOT_OUTCOME_INSTR, criteria=SHOT_OUTCOME_CRITERIA),
    "shot_outcome_rev": Choice(instructions=SHOT_OUTCOME_INSTR, criteria=_reversed(SHOT_OUTCOME_CRITERIA)),
}

RELEASE_INSTR = ("When the ball left the player's hands in the play described in `release`, was the "
                 "player trying to score or trying to pass?")
RELEASE_CRITERIA = {
    "shot_attempt": "Trying to score: the ball was thrown or tipped toward the basket",
    "pass": "Trying to get the ball to a teammate, including a lob for a teammate to catch near the rim",
    "unclear": "The cues in `release` do not show which one the player was trying to do",
}
B5_RELEASE = {
    "release_intent": Choice(instructions=RELEASE_INSTR, criteria=RELEASE_CRITERIA),
    "release_intent_rev": Choice(instructions=RELEASE_INSTR, criteria=_reversed(RELEASE_CRITERIA)),
}

# kind -> (state root key, questions, forward name, reversed name, committable labels)
B5_KINDS: dict[str, tuple[str, dict, str, str, tuple[str, str]]] = {
    "shot_outcome": ("shot", B5_SHOT_OUTCOME, "shot_outcome", "shot_outcome_rev", ("make", "miss")),
    "release_intent": ("release", B5_RELEASE, "release_intent", "release_intent_rev", ("shot_attempt", "pass")),
}


# ───────────────────────────── B6 cold_images (jev-1.14.0, images header) ─────────────────────────────
B6_RIM_FRAMES = {
    "rim_visible": Noul(instructions="Is the basketball rim clearly visible, not blocked by players, in `frames[2].frame`?"),
    "ball_above_rim_before": Noul(instructions="In `frames[0].frame`, is the basketball above the rim?"),
    "ball_in_net_after": Noul(instructions="In `frames[2].frame`, is the basketball inside the net, below the rim?"),
    "outcome_from_frames": Choice(
        instructions="Looking at `frames` in order, did the basketball go down through the hoop?",
        criteria={
            "make": "The ball goes down through the rim and net",
            "miss": "The ball does not go down through the rim",
            "unclear": "The frames do not show whether it went through",
        },
    ),
}
B6_SETUP = {
    "rim_present": Noul(instructions="Is there a basketball rim in `frame`?"),
    "rim_box": Box(instructions="the basketball rim nearest the camera in `frame`"),
    "scorebug_present": Noul(instructions="Is there an on-screen score graphic showing the game score in `frame`?"),
    "scorebug_box": Box(instructions="the on-screen score graphic in `frame`"),
}


# ───────────────────────────── hygiene helpers ─────────────────────────────


def question_texts(questions: Mapping[str, Any]) -> Iterator[str]:
    """Every instruction and criterion string (labels included) of a question mapping."""
    for q in questions.values():
        wire = q if isinstance(q, dict) else q.model_dump(mode="json", exclude_none=True)
        if isinstance(wire.get("instructions"), str):
            yield wire["instructions"]
        crit = wire.get("criteria")
        if isinstance(crit, Mapping):
            for label, desc in crit.items():
                if wire.get("type") == "choice":
                    yield label
                if isinstance(desc, str):
                    yield desc
        elif isinstance(crit, (list, tuple)):
            yield from (c for c in crit if isinstance(c, str))


def all_shipped_bundles() -> dict[str, dict]:
    """Every question set CourtVision can send, with the dynamic builders at their largest size."""
    return {
        "B1": B1_POSSESSION_END,
        "B2": b2_dead_ball_fan({"stat_note", "tendency", "storyline"}, MAX_STORYLINES),
        "B3": b3_dead_ball_coach({k: "" for k in CARD_KEYS}),
        "B4": b4_line_guard(MAX_GUARD_SENTENCES),
        "B5_shot": B5_SHOT_OUTCOME,
        "B5_release": B5_RELEASE,
        "B6_rim": B6_RIM_FRAMES,
        "B6_setup": B6_SETUP,
    }
