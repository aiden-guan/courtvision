"""PseudonymMap: per-game shuffled tokens and identity-tier rendering (D16, RESEARCH.md §4.1/§7A.7)."""

from __future__ import annotations

import re

from test_narration_core_support import make_roster

from courtvision.narration.pseudonym import PseudonymConfig, PseudonymMap, possessive
from courtvision.types import Roster, RosterEntry, Team


def test_tokens_unique_formatted_and_cover_roster() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=11)
    toks = [pm.token(e.player_id) for e in roster.entries]
    assert len(set(toks)) == len(toks)
    assert all(re.fullmatch(r"P(0[1-9]|[12]\d|30)", t) for t in toks)
    for e, t in zip(roster.entries, toks):
        assert pm.player(t) == e.player_id
        assert pm.player("{" + t + "}") == e.player_id
    assert pm.player("P99") is None


def test_tokens_are_shuffled_not_roster_or_jersey_order() -> None:
    roster = make_roster()
    seen = []
    for seed in range(5):
        pm = PseudonymMap(roster, seed=seed)
        nums = [int(pm.token(e.player_id)[1:]) for e in roster.entries]
        assert nums != sorted(nums)  # not handed out in roster order
        by_jersey = sorted(roster.entries, key=lambda e: (e.team.value, int(e.jersey)))
        assert [int(pm.token(e.player_id)[1:]) for e in by_jersey] != sorted(nums)
        seen.append(tuple(nums))
    assert len(set(seen)) == len(seen)  # per-game seeds give different maps


def test_same_seed_same_map() -> None:
    roster = make_roster()
    a, b = PseudonymMap(roster, seed=4), PseudonymMap(roster, seed=4)
    assert a.tokens() == b.tokens()


def test_provisional_ids_get_fresh_tokens_and_pool_grows() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=1)
    provisional = [f"home-t{i}" for i in range(40)]
    toks = [pm.token(p) for p in provisional]
    assert len(set(toks)) == len(toks)
    assert set(toks).isdisjoint(pm.token(e.player_id) for e in roster.entries)
    assert pm.token("home-t3") == toks[3]  # stable on reuse
    assert any(int(t[1:]) > 30 for t in toks)  # 12 roster + 40 provisional > 30


def test_render_identity_tiers() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=2)
    tok = pm.token("home-23")  # Riley Thorsen, jersey 23
    text = "{" + tok + "} buries the three!"
    assert pm.render(text, roster) == "Riley Thorsen buries the three!"  # missing conf => 1.0
    assert pm.render(text, roster, {"home-23": 0.97}) == "Riley Thorsen buries the three!"
    assert pm.render(text, roster, {"home-23": 0.90}, {"home-23": 0.96}) == "Riverside's number 23 buries the three!"
    assert pm.render(text, roster, {"home-23": 0.90}, {"home-23": 0.90}) == "Riverside buries the three!"
    assert pm.render(text, roster, {"home-23": 0.70}, {"home-23": 0.99}) == "Riverside buries the three!"
    # confidences may be keyed by token as well
    assert pm.render(text, roster, {tok: 0.85}, {tok: 0.99}) == "Riverside's number 23 buries the three!"


def test_render_boundaries_and_jersey_zero_strings() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=2)
    t0, t00 = pm.token("home-0"), pm.token("home-00")
    out = pm.render("{%s} to {%s}." % (t0, t00), roster, {"home-0": 0.8, "home-00": 0.8},
                    {"home-0": 0.95, "home-00": 0.95})
    assert out == "Riverside's number 0 to Riverside's number 00."
    assert pm.render("{%s}." % t0, roster, {"home-0": 0.95}) == "Avery Lindqvist."


def test_render_provisional_and_unknown() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=5)
    th, tu = pm.token("away-t17"), pm.token("unk-t5")
    assert pm.render("{%s} with the steal." % th, roster) == "Lakeport with the steal."
    assert pm.render("{%s} with the steal." % tu, roster) == "A player with the steal."
    assert pm.render("Steal by {%s}." % tu, roster) == "Steal by a player."
    assert pm.render("Steal by {P99}.", roster) == "Steal by a player."


def test_possessive_and_team_tier_with_plural_name() -> None:
    assert possessive("Hawks") == "Hawks'"
    assert possessive("Home") == "Home's"
    roster = Roster([RosterEntry("home-4", Team.HOME, "4", "Sam Pelletier")], {Team.HOME: "Hawks", Team.AWAY: "Owls"})
    pm = PseudonymMap(roster, seed=0, config=PseudonymConfig())
    tok = pm.token("home-4")
    assert pm.render("{%s} scores." % tok, roster, {tok: 0.9}, {tok: 0.99}) == "Hawks' number 4 scores."


def test_pseudonymise_replaces_names_and_surnames() -> None:
    roster = make_roster()
    pm = PseudonymMap(roster, seed=9)
    text = "Riley Thorsen to Okafor, then Achterberg again."
    out = pm.pseudonymise(text, roster)
    assert "Thorsen" not in out and "Okafor" not in out and "Achterberg" not in out
    assert out == "{%s} to {%s}, then {%s} again." % (pm.token("home-23"), pm.token("home-3"), pm.token("away-0"))
    assert pm.render(out, roster) == "Riley Thorsen to Jordan Okafor, then Devon Achterberg again."
