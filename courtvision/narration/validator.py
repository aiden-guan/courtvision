"""Deterministic grounding validator for narration lines (narration.md §3.3, RESEARCH.md §2.6).

Every sentence (LLM or template, still in ``{Pnn}`` token form) is checked against its FactPacket
before it may be rendered and spoken:

* tokens: every ``{Pnn}`` is in ``allowed_tokens`` (bare ``P07`` or other ``{...}`` placeholders fail);
* numbers: digits, number words (up to ninety-nine, plus "hundred"), ordinals, ``4:32`` clocks,
  ``3-for-5`` / ``3 of 5`` shooting lines and percentages must all appear in ``allowed_numbers``;
* claims lexicon: career / season / record / all-time / streak / injury / "first time" /
  "tonight's high" ... only when a ``derived``, ``storylines`` or ``labels`` phrase supports them;
* score consistency: every ``A-B`` pair is the score (or a code-computed run), lead/trail/tie
  phrasing matches the score;
* no future-tense predictions; per-channel word caps; terminal punctuation;
* repetition: token Jaccard >= 0.8 against any ``recent_lines`` entry fails;
* hedge-only packets: no outcome words or point values;
* unknown capitalised names in mid-sentence (team names and a small allowlist pass).

All problems are returned, not just the first. Target < 5 ms per sentence.

The module also exports the text utilities shared by ``facts`` and ``templates`` (number words,
number extraction, word counting, Jaccard).
"""

from __future__ import annotations

import functools
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- tokens

TOKEN_RE = re.compile(r"\{(P\d{2,3})\}")
_BARE_TOKEN_RE = re.compile(r"(?<![\w{])P\d{2,3}(?![\w}])")
_PLACEHOLDER_RE = re.compile(r"\{[^{}]*\}")
_ANY_TOKEN_RE = re.compile(r"\{P\d{2,3}\}|(?<![\w{])P\d{2,3}(?![\w}])")

# --------------------------------------------------------------------------- number words

_UNIT_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS_WORDS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty", 6: "sixty", 7: "seventy", 8: "eighty", 9: "ninety"}
_ORD_UNIT_WORDS = (
    "zeroth first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth "
    "thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth"
).split()
_ORD_TENS_WORDS = {2: "twentieth", 3: "thirtieth", 4: "fortieth", 5: "fiftieth", 6: "sixtieth",
                   7: "seventieth", 8: "eightieth", 9: "ninetieth"}

UNITS: dict[str, int] = {w: i for i, w in enumerate(_UNIT_WORDS)}
TENS: dict[str, int] = {w: k * 10 for k, w in _TENS_WORDS.items()}
ORD_UNITS: dict[str, int] = {w: i for i, w in enumerate(_ORD_UNIT_WORDS) if i > 0}
ORD_TENS: dict[str, int] = {w: k * 10 for k, w in _ORD_TENS_WORDS.items()}


def int_to_words(n: int) -> str:
    """0..999 as English words ("forty-two", "one hundred five"); larger values as digits."""
    n = int(n)
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _UNIT_WORDS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        return _TENS_WORDS[tens] + ("" if unit == 0 else "-" + _UNIT_WORDS[unit])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _UNIT_WORDS[hundreds] + " hundred" + ("" if rest == 0 else " " + int_to_words(rest))
    return str(n)


def ordinal_words(n: int) -> str:
    """1..99 as ordinal words ("first", "twenty-third"); other values as digits + suffix."""
    n = int(n)
    if 1 <= n < 20:
        return _ORD_UNIT_WORDS[n]
    if 20 <= n < 100:
        tens, unit = divmod(n, 10)
        if unit == 0:
            return _ORD_TENS_WORDS[tens]
        return _TENS_WORDS[tens] + "-" + _ORD_UNIT_WORDS[unit]
    return ordinal_digits(n)


def ordinal_digits(n: int) -> str:
    """4 -> "4th", 21 -> "21st"."""
    n = int(n)
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# --------------------------------------------------------------------------- number extraction


@dataclass(frozen=True, slots=True)
class NumberMention:
    """One number found in text. ``key`` is canonical: card:42, ord:4, clock:4:32, pair:3/5,
    dash:42-40, pct:60. ``alt_keys`` are accepted instead of ``key`` ("3 of 5" may also be two
    plain numbers)."""

    key: str
    text: str
    alt_keys: tuple[str, ...] = ()


def _fmt_num(s: str) -> str:
    v = float(s)
    return str(int(v)) if v.is_integer() else f"{v:g}"


_CLOCK_RE = re.compile(r"(?<![\d:.])(\d{1,2}):([0-5]\d)(?:\.\d)?(?![\d:])")
_PCT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)", re.I)
_WORD_NUM = r"[a-z]+(?:-[a-z]+)?"
_PAIR_HYPHEN_RE = re.compile(rf"(?<![\w-])(\d+|{_WORD_NUM})-for-(\d+|{_WORD_NUM})(?![\w-])", re.I)
# Lookaheads reject a decimal continuation ("3.5") but not a sentence-final period ("42-40.").
_PAIR_SPACE_RE = re.compile(r"(?<![\w.])(\d+)\s+(for|of)\s+(\d+)(?!\w|\.\d)", re.I)
_PAIR_WORD_OF_RE = re.compile(rf"(?<![\w-])({_WORD_NUM})\s+of\s+({_WORD_NUM})(?![\w-])", re.I)
_DASH_RE = re.compile(r"(?<![\w.\-])(\d{1,3})\s*[-–—]\s*(\d{1,3})(?![\w\-]|\.\d)")
_ORD_DIGIT_RE = re.compile(r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b", re.I)
_DIGITS_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_WORD_RE = re.compile(r"[a-z]+(?:['’-][a-z]+)*")

# Hyphen compounds whose number part is idiomatic, not a quantity ("one-on-one", "two-handed").
_IDIOM_PARTS = frozenset({"on", "handed", "hand", "legged", "footed", "sided", "way", "timer", "time",
                          "chance", "effort", "and", "off", "liner", "guessing", "string", "unit"})
# "one" used as a pronoun ("that one", "no one", "and one"), not a count.
_ONE_PREV_IDIOM = frozenset({"no", "that", "this", "the", "which", "every", "each", "any", "another", "some",
                             "big", "only", "last", "next", "other", "little", "good", "tough", "nice",
                             "great", "and", "a", "such", "what", "quick", "easy", "hard", "right",
                             "wrong", "first", "late", "early"})
# "second" as a time unit or idiom ("a second", "second chance").
_SECOND_PREV_IDIOM = frozenset({"a", "one", "per", "split", "every", "each", "few", "half"})
_SECOND_NEXT_IDIOM = frozenset({"chance", "chances", "effort", "efforts", "look", "wind", "unit", "string", "guessing"})
_FIRST_NEXT_IDIOM = frozenset({"step", "look", "glance"})
_FIRST_PREV_IDIOM = frozenset({"at"})


def _word_value(w: str) -> tuple[int, bool] | None:
    """Single (possibly hyphenated) number word -> (value, is_ordinal)."""
    w = w.lower()
    if w in UNITS:
        return UNITS[w], False
    if w in TENS:
        return TENS[w], False
    if w in ORD_UNITS:
        return ORD_UNITS[w], True
    if w in ORD_TENS:
        return ORD_TENS[w], True
    if "-" in w:
        a, _, b = w.partition("-")
        if a in TENS and b in UNITS and 1 <= UNITS[b] <= 9:
            return TENS[a] + UNITS[b], False
        if a in TENS and b in ORD_UNITS and 1 <= ORD_UNITS[b] <= 9:
            return TENS[a] + ORD_UNITS[b], True
    return None


def _num_or_word(s: str) -> int | float | None:
    if s[0].isdigit():
        return float(s) if "." in s else int(s)
    v = _word_value(s)
    return v[0] if v is not None and not v[1] else None


def _mask(chars: list[str], start: int, end: int) -> None:
    for i in range(start, end):
        chars[i] = " "


def _strip_tokens(text: str) -> str:
    return _ANY_TOKEN_RE.sub(lambda m: " " * len(m.group(0)), text)


def extract_numbers(text: str) -> list[NumberMention]:
    """All number mentions in ``text`` (tokens such as ``{P07}`` are ignored)."""
    s = _strip_tokens(text)
    chars = list(s)
    out: list[NumberMention] = []

    def run(pattern: re.Pattern[str], handle: Any) -> None:
        cur = "".join(chars)
        for m in pattern.finditer(cur):
            mention = handle(m)
            if mention is not None:
                out.append(mention)
                _mask(chars, m.start(), m.end())

    run(_CLOCK_RE, lambda m: NumberMention(f"clock:{int(m.group(1))}:{m.group(2)}", m.group(0)))
    run(_PCT_RE, lambda m: NumberMention(f"pct:{_fmt_num(m.group(1))}", m.group(0)))

    def pair(m: re.Match[str]) -> NumberMention | None:
        a, b = _num_or_word(m.group(1)), _num_or_word(m.group(2))
        if a is None or b is None:
            return None
        return NumberMention(f"pair:{a}/{b}", m.group(0))

    run(_PAIR_HYPHEN_RE, pair)

    def pair_space(m: re.Match[str]) -> NumberMention:
        a, b = int(m.group(1)), int(m.group(3))
        alt = (f"card:{a}", f"card:{b}") if m.group(2).lower() == "of" else ()
        return NumberMention(f"pair:{a}/{b}", m.group(0), alt)

    run(_PAIR_SPACE_RE, pair_space)

    def pair_word_of(m: re.Match[str]) -> NumberMention | None:
        a, b = _num_or_word(m.group(1)), _num_or_word(m.group(2))
        if a is None or b is None:
            return None
        return NumberMention(f"pair:{a}/{b}", m.group(0), (f"card:{a}", f"card:{b}"))

    run(_PAIR_WORD_OF_RE, pair_word_of)
    run(_DASH_RE, lambda m: NumberMention(f"dash:{int(m.group(1))}-{int(m.group(2))}", m.group(0)))
    run(_ORD_DIGIT_RE, lambda m: NumberMention(f"ord:{int(m.group(1))}", m.group(0)))
    run(_DIGITS_RE, lambda m: NumberMention(f"card:{_fmt_num(m.group(1))}", m.group(0)))
    out.extend(_word_numbers("".join(chars).lower()))
    return out


def _word_numbers(s: str) -> list[NumberMention]:
    words = [w.split("'")[0].split("’")[0] for w in _WORD_RE.findall(s)]
    out: list[NumberMention] = []
    i = 0
    while i < len(words):
        w = words[i]
        prev = words[i - 1] if i > 0 else ""
        nxt = words[i + 1] if i + 1 < len(words) else ""
        if "-" in w and _word_value(w) is None:
            out.extend(_compound(w))
            i += 1
            continue
        if w == "a" and nxt == "hundred":
            value, used = _hundreds(words, i + 1, 1)
            out.append(NumberMention(f"card:{value}", " ".join(words[i:i + 1 + used])))
            i += 1 + used
            continue
        wv = _word_value(w)
        if wv is None:
            i += 1
            continue
        value, is_ord = wv
        if is_ord:
            if _ordinal_is_idiom(w, prev, nxt):
                i += 1
                continue
            out.append(NumberMention(f"ord:{value}", w))
            i += 1
            continue
        used = 1
        if w in TENS and nxt in UNITS and 1 <= UNITS[nxt] <= 9:
            value += UNITS[nxt]
            used = 2
        elif 1 <= value <= 9 and nxt == "hundred":
            value, extra = _hundreds(words, i + 1, value)
            used = 1 + extra
        if value == 1 and used == 1 and (prev in _ONE_PREV_IDIOM or nxt == "another"):
            i += 1
            continue
        out.append(NumberMention(f"card:{value}", " ".join(words[i:i + used])))
        i += used
    return out


def _hundreds(words: Sequence[str], j: int, mult: int) -> tuple[int, int]:
    """Parse "<mult> hundred [and] [rest]" starting at ``words[j] == "hundred"``; returns (value, used)."""
    value, used = mult * 100, 1
    k = j + 1
    if k < len(words) and words[k] == "and":
        k += 1
    if k < len(words):
        rest = _word_value(words[k])
        if rest is not None and not rest[1] and rest[0] < 100:
            value += rest[0]
            used = k - j + 1
    return value, used


def _compound(w: str) -> list[NumberMention]:
    parts = w.split("-")
    if any(p in _IDIOM_PARTS for p in parts):
        return []
    if "for" in parts or "of" in parts:
        return []
    # leading number words ("three-pointer", "twenty-four-second") carry the quantity
    if len(parts) >= 2:
        two = _word_value(parts[0] + "-" + parts[1])
        if two is not None:
            key = f"{'ord' if two[1] else 'card'}:{two[0]}"
            return [NumberMention(key, w)]
    first = _word_value(parts[0])
    if first is None:
        return []
    value, is_ord = first
    if is_ord and parts[1:] and parts[1] in _SECOND_NEXT_IDIOM:
        return []
    return [NumberMention(f"{'ord' if is_ord else 'card'}:{value}", w)]


def _ordinal_is_idiom(w: str, prev: str, nxt: str) -> bool:
    if w == "second":
        return prev in _SECOND_PREV_IDIOM or nxt in _SECOND_NEXT_IDIOM
    if w == "first":
        return nxt in _FIRST_NEXT_IDIOM or prev in _FIRST_PREV_IDIOM
    return False


@functools.lru_cache(maxsize=256)
def _keys_cached(strings: tuple[str, ...]) -> frozenset[str]:
    keys: set[str] = set()
    for s in strings:
        for m in extract_numbers(s):
            keys.add(m.key)
    return frozenset(keys)


def number_keys(strings: Iterable[str]) -> frozenset[str]:
    """Canonical keys of every number mentioned in ``strings`` (cached)."""
    return _keys_cached(tuple(str(s) for s in strings))


def number_forms(value: int | float, ordinal: bool = False) -> list[str]:
    """Display forms of one number for ``allowed_numbers``: digits and words (+ ordinals)."""
    if isinstance(value, float) and not value.is_integer():
        return [f"{value:g}"]
    n = int(value)
    forms = [str(n)]
    if 0 <= n < 1000:
        forms.append(int_to_words(n))
    if ordinal and n >= 1:
        forms += [ordinal_digits(n), ordinal_words(n)]
    return forms


# --------------------------------------------------------------------------- words


_WORDS_FOR_JACCARD = re.compile(r"\{p\d{2,3}\}|[a-z0-9]+(?:['’-][a-z0-9]+)*")


def count_words(text: str) -> int:
    """Whitespace-separated words; a ``{P07}`` token counts as one word."""
    return len(text.split())


def word_set(text: str) -> frozenset[str]:
    return frozenset(_WORDS_FOR_JACCARD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity of two lines (lower-cased words; tokens count as words)."""
    sa, sb = word_set(a), word_set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def split_sentences(text: str) -> list[str]:
    """Sentences ending in . ! ? or an ellipsis (the remainder counts as a sentence)."""
    parts = re.findall(r"[^.!?…]+(?:\.\.\.|[.!?…]+)?", text)
    return [p.strip() for p in parts if p.strip()]


# --------------------------------------------------------------------------- lexicons

_STREAK_RE = (r"\bstreak\w*\b|\bin a row\b|\bconsecutive\b|\bunanswered\b"
              r"|\bstraight\s+(?:makes?|points|buckets?|baskets?|shots?|stops?|games?|wins?|threes?|scores?|field goals?)\b"
              r"|\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|second|third|fourth|fifth"
              r"|sixth|seventh|eighth|ninth|tenth)\s+straight\b")

CLAIM_PATTERNS: dict[str, str] = {
    "career": r"\bcareer\b",
    "season": r"\bseason\w*\b|\bthis year\b|\blast year\b",
    "record": r"\brecords?\b",
    "all-time": r"\ball[- ]time\b",
    "franchise": r"\bfranchise\b|\bhistory\b|\bhistoric\w*\b",
    "streak": _STREAK_RE,
    "injury": r"\binjur\w*\b|\bhurt\b|\bankle\b|\bknee\b|\bhamstring\b|\bconcussion\b|\bsprain\w*\b|\blimp\w*\b",
    "first time": r"\bfirst time\b|\bfor the first\b",
    "tonight's high": r"\btonight'?s high\b|\bgame[- ]high\b|\bpersonal best\b|\bbest ever\b|\bhigh of the night\b|\bcareer[- ]high\b",
    "never/ever": r"\bnever\b|\bever\b",
    "average": r"\baverag\w*\b|\bper game\b",
    "past games": r"\blast (?:game|night|week|season|meeting|time out)\b|\byesterday\b|\bprevious game\b",
    "accolade": r"\bmvp\b|\ball-star\b|\brookie\b|\bveteran\b|\bdraft\w*\b|\btraded?\b|\bcontract\b|\bscholarship\b|\brecruit\w*\b",
    "milestone": r"\bmilestone\b",
    "discipline": r"\beject\w*\b|\bsuspen\w*\b|\bflagrant\b|\btechnical\b",
    "play type": r"\bpick[- ]and[- ]roll\b|\bisolation\b|\bpost[- ]up\b|\bfast break\b|\btransition\b|\bzone defense\b",
}
_CLAIM_RES = {name: re.compile(p, re.I) for name, p in CLAIM_PATTERNS.items()}

_FUTURE_VERBS = (r"win|wins|lose|loses|score|make|miss|hit|beat|take\s+over|come\s+back|pull\s+away|run\s+away"
                 r"|close\s+(?:it|out)|clinch|seal|foul\s+out|finish|get|need|be|have|go\s+on|cover|lead|force")
FUTURE_RE = re.compile(
    r"(?:\bwill|\bwon'?t|'ll|\bshall|\bgonna|\bgoing\s+to|\bbound\s+to|\bsure\s+to|\bdestined\s+to"
    r"|\bon\s+(?:their|his|her|its)\s+way\s+to)\s+(?:\w+\s+){0,2}?(?:" + _FUTURE_VERBS + r")\b",
    re.I,
)

# Outcome words and point values: forbidden while a packet is hedge-only (RESEARCH.md §7A.2). The noun
# "the score" / "score check" (a confirmed score line) and number words inside a margin or a
# compound ("by three", "twenty-three") are not outcomes; "for three", "a three", "threes" are.
OUTCOME_RE = re.compile(
    r"\b(?:no good|good|makes?|made|miss(?:es|ed)?|scores|(?<!the )score(?!\s*(?:check\b|:|is\b|was\b|stands\b))"
    r"|scored|scoring|counts?|counted|buckets?|baskets?"
    r"|blocks?|blocked|rejects?|rejected|denied|swats?|stuff(?:s|ed)?|steals?|stole|stolen|picks? it off|picks? off"
    r"|turnovers?|turns? it over|giveaway|and[- ]one|drains?|buries|bury|hits|sinks?|nails?|knocks? (?:it )?down"
    r"|splash|swish|throws? it down|slams?|jams?|flush(?:es)?|finish(?:es)?|converts?|rims? out|in and out"
    r"|off the rim|airball|bricks?|clanks?|points?|pointers?|(?<!-)(?<!by )(?<!up )(?<!down )(?:threes?|twos?)"
    r"|[123]-?pointers?|got it|cashes"
    r"|connects?|dimes?|assists?)\b",
    re.I,
)

# One spoken number (digits or words, up to "nine hundred ninety-nine") inside a score phrase.
_SUB100 = (r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six"
           r"|seven|eight|nine))?|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen"
           r"|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen)")
_SCORE_NUM = (rf"(?:\d+|(?:a|one|two|three|four|five|six|seven|eight|nine)\s+hundred(?:\s+(?:and\s+)?{_SUB100})?"
              rf"|{_SUB100})")
_TO_PAIR_RE = re.compile(rf"(?<![\w.-])({_SCORE_NUM})\s+to\s+({_SCORE_NUM})(?![\w-])", re.I)

_LEAD_VERB_BY_RE = re.compile(
    r"\b(?:leads?|leading|up|ahead|in front|trails?|trailing|down|behind)\s+by\s+(" + _SCORE_NUM + r")(?![\w-])", re.I)
# "a two-point game", "a 5-point lead": the number is the current margin.
_MARGIN_NOUN_RE = re.compile(
    r"\b(\d+|[a-z]+(?:-[a-z]+)?)[- ]points?\s+(?:game|lead|deficit|margin|edge|cushion|hole)\b", re.I)
_TIE_RE = re.compile(r"\b(?:tied|tie game|ties it|all square|even at|level at|knotted|dead even)\b", re.I)


# --------------------------------------------------------------------------- validator


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str] = field(default_factory=list)


@dataclass
class ValidatorConfig:
    """Grounding-validator settings (narration.md §3.3 / §9.2; RESEARCH.md §2.6)."""

    # Word caps per channel (narration.md §9.2, design proposals): live calls <= 12 (score/run update
    # cap), colour <= 35 per line (timeout colour), coach cards <= 12, AD <= 6 per call (score line
    # exempt, RESEARCH.md §7A.8), corrections <= 12. unverified (tune on labelled games).
    max_words: dict[str, int] = field(default_factory=lambda: {
        "play_by_play": 12, "color": 35, "coach": 12, "accessibility": 6, "correction": 12})
    default_channel: str = "color"  # LLM lines are colour unless the caller says otherwise
    repetition_jaccard: float = 0.8  # token Jaccard >= this vs any recent line fails (spec); tuned
    repetition_skip_channels: tuple[str, ...] = ("accessibility",)  # AD uses a fixed lexicon by design
    require_terminal_punct: bool = True  # a missing mark suggests max_tokens truncation (narration.md §3.3)
    check_entities: bool = True  # unknown capitalised mid-sentence words fail
    # Capitalised words that are not names (abbreviations used on coach cards, interjection "I").
    entity_allowlist: tuple[str, ...] = ("I", "Correction", "FG", "FT", "PTS", "REB", "AST", "STL", "BLK",
                                         "TOV", "PF", "OT", "PPP", "eFG", "ORB", "DRB", "Q1", "Q2", "Q3", "Q4")
    correction_prefix: str = "correction"  # correction-channel lines start with "Correction:" (narration.md §9.2)


class GroundingValidator:
    """Checks one line (``{Pnn}`` token form) against its FactPacket. Stateless and fast."""

    def __init__(self, config: ValidatorConfig | None = None) -> None:
        self.config = config or ValidatorConfig()
        self._allow_lower = frozenset(w.lower() for w in self.config.entity_allowlist)

    def validate(self, text: str, facts: Mapping[str, Any], channel: str | None = None) -> ValidationResult:
        """Return every grounding problem of ``text``; ``ok`` when there are none."""
        channel = channel or str(facts.get("channel") or self.config.default_channel)
        problems: list[str] = []
        if not text or not text.strip():
            return ValidationResult(False, ["empty line"])
        support = _support_text(facts)
        problems += self._check_tokens(text, facts)
        problems += self._check_numbers(text, facts)
        problems += self._check_score(text, facts, support)
        problems += self._check_claims(_mask_team_names(text, facts), _mask_team_names(support, facts))
        problems += self._check_future(text)
        problems += self._check_length(text, facts, channel)
        problems += self._check_repetition(text, facts, channel)
        problems += self._check_form(text, channel)
        problems += self._check_entities(text, facts)
        if facts.get("hedge_only"):
            problems += [f"outcome word while hedge_only: {m.group(0)!r}"
                         for m in OUTCOME_RE.finditer(_strip_tokens(text))]
        if problems:
            log.debug("validator rejected %r: %s", text, problems)
        return ValidationResult(not problems, problems)

    # ------------------------------------------------------------------ checks

    @staticmethod
    def _check_tokens(text: str, facts: Mapping[str, Any]) -> list[str]:
        allowed = {str(t).strip("{}") for t in facts.get("allowed_tokens") or ()}
        problems = [f"unknown token {{{t}}}" for t in TOKEN_RE.findall(text) if t not in allowed]
        problems += [f"malformed token {m.group(0)!r}" for m in _BARE_TOKEN_RE.finditer(text)]
        problems += [f"unresolved placeholder {m.group(0)!r}" for m in _PLACEHOLDER_RE.finditer(text)
                     if not TOKEN_RE.fullmatch(m.group(0))]
        return problems

    @staticmethod
    def _check_numbers(text: str, facts: Mapping[str, Any]) -> list[str]:
        allowed = number_keys(facts.get("allowed_numbers") or ())
        problems = []
        for m in extract_numbers(text):
            if m.key.startswith("dash:"):
                continue  # A-B pairs are checked by the score-consistency rule
            if m.key in allowed or (m.alt_keys and all(k in allowed for k in m.alt_keys)):
                continue
            problems.append(f"number not in facts: {m.text!r}")
        return problems

    def _check_score(self, text: str, facts: Mapping[str, Any], support: str) -> list[str]:
        score = facts.get("score") or {}
        h, a = score.get("home"), score.get("away")
        pairs: set[tuple[int, int]] = set()
        if h is not None and a is not None:
            pairs |= {(int(h), int(a)), (int(a), int(h))}
        run = facts.get("run") or {}
        if run.get("points") is not None and run.get("opp_points") is not None:
            pairs.add((int(run["points"]), int(run["opp_points"])))
        for k in number_keys([support, *(str(s) for s in facts.get("allowed_numbers") or ())]):
            if k.startswith("dash:"):
                x, _, y = k[5:].partition("-")
                pairs.add((int(x), int(y)))
        problems = []
        unspeakable = score.get("speakable") is False  # unconfirmed score from unverified outcomes
        shown = score.get("text") or (f"{h}-{a}" if h is not None else "unknown")
        for m in extract_numbers(text):
            if m.key.startswith("dash:"):
                x, _, y = m.key[5:].partition("-")
                if (int(x), int(y)) not in pairs:
                    problems.append(f"score inconsistent: {m.text!r} (score {shown})")
                elif unspeakable and h is not None and {int(x), int(y)} == {int(h), int(a)}:
                    problems.append(f"score not speakable (unconfirmed): {m.text!r}")
        if h is None or a is None:
            return problems
        clean = _strip_tokens(text)
        problems += _check_score_phrases(clean, facts, int(h), int(a), pairs, shown)
        line_re = _score_line_re(facts)
        if unspeakable and line_re is not None and line_re.search(clean):
            problems.append("score not speakable (unconfirmed)")
        margin, leader = abs(int(h) - int(a)), score.get("leader")
        for rx in (_LEAD_VERB_BY_RE, _MARGIN_NOUN_RE):
            for m in rx.finditer(clean):
                n = _spoken_value(m.group(1))
                if n is not None and n != margin:
                    problems.append(f"margin inconsistent: {m.group(0)!r} (margin {margin})")
        problems += self._check_team_lead(clean, facts, margin, leader)
        if margin != 0:
            problems += [f"tie claimed but margin is {margin}: {m.group(0)!r}" for m in _TIE_RE.finditer(clean)]
        return problems

    @staticmethod
    def _check_team_lead(clean: str, facts: Mapping[str, Any], margin: int, leader: str | None) -> list[str]:
        problems: list[str] = []
        for name in _team_names(facts):
            margin_re, lead_re, trail_re = _team_regexes(name)
            is_leader = leader is not None and leader.lower() == name.lower()
            for m in margin_re.finditer(clean):
                n = _spoken_value(m.group(1))
                if n is not None and (not is_leader or n != margin):
                    problems.append(f"lead inconsistent: {m.group(0)!r} (leader {leader}, margin {margin})")
            if not is_leader:
                problems += [f"leader inconsistent: {m.group(0)!r} (leader {leader})" for m in lead_re.finditer(clean)]
            if leader is None or is_leader:
                problems += [f"trailer inconsistent: {m.group(0)!r} (leader {leader})" for m in trail_re.finditer(clean)]
        return problems

    @staticmethod
    def _check_claims(text: str, support: str) -> list[str]:
        problems = []
        for name, rx in _CLAIM_RES.items():
            hit = rx.search(text)
            if hit and not rx.search(support):
                problems.append(f"unsupported claim ({name}): {hit.group(0)!r}")
        return problems

    @staticmethod
    def _check_future(text: str) -> list[str]:
        return [f"prediction: {m.group(0)!r}" for m in FUTURE_RE.finditer(text)]

    def _check_length(self, text: str, facts: Mapping[str, Any], channel: str) -> list[str]:
        cap = facts.get("max_words") or self.config.max_words.get(channel)
        if not cap:
            return []
        if channel == "accessibility":
            score_line = _score_line_re(facts)
            n = sum(count_words(s) for s in split_sentences(text) if not (score_line and score_line.fullmatch(s)))
        else:
            n = count_words(text)
        return [f"too long: {n} words > {cap} ({channel})"] if n > int(cap) else []

    def _check_repetition(self, text: str, facts: Mapping[str, Any], channel: str) -> list[str]:
        if channel in self.config.repetition_skip_channels:
            return []
        thr = self.config.repetition_jaccard
        for line in facts.get("recent_lines") or ():
            if jaccard(text, str(line)) >= thr:
                return [f"repeats recent line: {line!r}"]
        return []

    def _check_form(self, text: str, channel: str) -> list[str]:
        problems = []
        stripped = text.rstrip().rstrip("\"'”")
        if self.config.require_terminal_punct and stripped[-1:] not in (".", "!", "?", "…"):
            problems.append("no terminal punctuation")
        if channel == "correction" and not text.lstrip().lower().startswith(self.config.correction_prefix):
            problems.append("correction line must start with 'Correction:'")
        return problems

    def _check_entities(self, text: str, facts: Mapping[str, Any]) -> list[str]:
        if not self.config.check_entities:
            return []
        known = set(self._allow_lower)
        for name in _team_names(facts):
            known.update(w.lower() for w in re.findall(r"[A-Za-z][\w'’-]*", name))
        problems = []
        clean = _ANY_TOKEN_RE.sub("tok", text)  # a word, so a name right after a token is mid-sentence
        for m in re.finditer(r"[A-Z][A-Za-z'’-]*", clean):
            start = m.start()
            if start > 0 and (clean[start - 1].isalnum() or clean[start - 1] in "'’-"):
                continue
            if _at_sentence_start(clean, start):
                continue
            word = re.sub(r"(?:'s|’s|'|’)$", "", m.group(0))
            if word.lower() not in known:
                problems.append(f"unknown name {word!r}")
        return problems


# --------------------------------------------------------------------------- helpers


def _support_text(facts: Mapping[str, Any]) -> str:
    """Code-made phrases that may license claims (derived, storylines, soft labels)."""
    parts: list[str] = []
    for key in ("derived", "storylines"):
        v = facts.get(key) or []
        parts.extend(str(x) for x in (v.values() if isinstance(v, Mapping) else v))
    labels = facts.get("labels") or {}
    if isinstance(labels, Mapping):
        parts.extend(str(v) for v in labels.values())
    return " | ".join(parts)


def _mask_team_names(text: str, facts: Mapping[str, Any]) -> str:
    """Team names may contain lexicon words ("... Record Hall"); claims are checked without them."""
    for name in _team_names(facts):
        text = re.sub(re.escape(name), "team", text, flags=re.I)
    return text


def _spoken_value(s: str) -> int | None:
    vals = [m.key for m in extract_numbers(s) if m.key.startswith("card:")]
    return int(float(vals[0][5:])) if len(vals) == 1 else None


def _check_score_phrases(clean: str, facts: Mapping[str, Any], h: int, a: int, pairs: set[tuple[int, int]],
                         shown: str) -> list[str]:
    """"<Team> N, <Team> M" must give each team its own score (or the run's split), and an
    "N to M" pair that names a score number must be the score (or a code-made pair)."""
    problems: list[str] = []
    names = _team_names(facts)
    if len(names) == 2:
        home, away = names[0].lower(), names[1].lower()
        allowed = [{home: h, away: a}]
        run = facts.get("run") or {}
        run_team = str(run.get("team") or "").lower()
        if run_team in (home, away) and run.get("points") is not None and run.get("opp_points") is not None:
            allowed.append({run_team: int(run["points"]), (away if run_team == home else home): int(run["opp_points"])})
        for m in _score_pair_cached(tuple(names)).finditer(clean):
            v1, v2 = _spoken_value(m.group(2)), _spoken_value(m.group(4))
            if v1 is None or v2 is None:
                continue
            t1, t2 = m.group(1).lower(), m.group(3).lower()
            if t1 == t2 or {t1: v1, t2: v2} not in allowed:
                problems.append(f"score inconsistent: {m.group(0)!r} (score {shown})")
    for m in _TO_PAIR_RE.finditer(clean):
        v1, v2 = _spoken_value(m.group(1)), _spoken_value(m.group(2))
        if v1 is not None and v2 is not None and {v1, v2} & {h, a} and (v1, v2) not in pairs:
            problems.append(f"score inconsistent: {m.group(0)!r} (score {shown})")
    return problems


@functools.lru_cache(maxsize=64)
def _score_pair_cached(names: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"({alt})\s+({_SCORE_NUM})\s*,\s*({alt})\s+({_SCORE_NUM})(?![\w-])", re.I)


def _team_names(facts: Mapping[str, Any]) -> list[str]:
    teams = facts.get("teams") or {}
    names = []
    for side in ("home", "away"):
        t = teams.get(side) if isinstance(teams, Mapping) else None
        name = t.get("name") if isinstance(t, Mapping) else None
        if name:
            names.append(str(name))
    return names


@functools.lru_cache(maxsize=64)
def _team_regexes(name: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    """(margin, lead, trail) patterns for one team name, compiled once."""
    esc = re.escape(name)
    margin = re.compile(esc + r"(?:'s?)?\s+(?:is\s+|are\s+|now\s+)?(?:by|up|up by|leads? by|ahead by|in front by"
                        r"|on top by)\s+(" + _SCORE_NUM + r")(?![\w-])(?!\s*[-–—]\d)", re.I)
    # "winning"/"losing" only about the game ("Lakeport is winning."), not "winning the boards".
    game = r"(?=\s*(?:[.,!?;]|$)|\s+(?:by|this one|this game|the game|it|now|right now)\b)"
    lead = re.compile(esc + r"(?:'s?)?\s+(?:leads|lead|leading|is up|are up|up|now up|is ahead|are ahead|ahead|in front"
                      r"|on top|takes the lead|take the lead|goes ahead|go ahead|(?:has|have|holds?|with|in) the lead"
                      r"|(?:is|are|now)\s+(?:in front|on top|leading))\b|" + esc + r"\s+(?:is |are )?winning" + game,
                      re.I)
    trail = re.compile(esc + r"\s+(?:trails|trail|trailing|is down|are down|down|is behind|are behind|behind"
                       r"|(?:is|are)\s+trailing)\b|" + esc + r"\s+(?:is |are )?losing" + game, re.I)
    return margin, lead, trail


def _score_line_re(facts: Mapping[str, Any]) -> re.Pattern[str] | None:
    names = _team_names(facts)
    return _score_line_cached(tuple(names)) if len(names) == 2 else None


@functools.lru_cache(maxsize=64)
def _score_line_cached(names: tuple[str, ...]) -> re.Pattern[str]:
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(rf"(?:{alt}) \d+, (?:{alt}) \d+[.!]?", re.I)


def _at_sentence_start(s: str, i: int) -> bool:
    j = i - 1
    while j >= 0 and s[j] in " \t\"'“(":
        j -= 1
    return j < 0 or s[j] in ".!?:;…"
