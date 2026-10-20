# Rules overrides

State associations, tournaments, AAU events and rec leagues change a few rules of a base ruleset.
An override is a partial `RuleSpec` YAML that the operator picks at setup:

```python
from courtvision.geometry.rules import rules_for

rules = rules_for("nfhs", ["no_shot_clock"])            # a file in this folder, by name
rules = rules_for("nfhs", ["/path/to/spring_aau.yaml"])  # or an explicit path
```

Base rulesets live one folder up (`nba`, `fiba`, `ncaa_m`, `ncaa_w`, `nfhs`). AAU and adult rec
default to `nfhs` plus an override, because no primary AAU source was retrieved (RESEARCH.md §7A.13).

## Merge rules

* Overrides apply in list order; later files win.
* Mappings merge key by key (`timeouts: {count: 2}` keeps every other timeout key).
* Anything else replaces: scalars, lists and `null`. `shot_clock_s: null` means no shot clock,
  and `running_clock: null` turns a running clock off.
* Unknown keys are rejected with the file name, at the top level and inside `running_clock`,
  `timeouts` (and `timeouts.media`), `bonus`, `tech_ft`, `clock_stops_after_fg` and `intervals`.
  A typo fails loudly instead of being ignored.
* Set `ruleset:` to give a custom format its own id (it shows on the dashboard). Leave it out for
  a state variant of a base ruleset.
* `halves: true` requires `n_periods: 2`. `shot_clock_reset_s` must be `null` when
  `shot_clock_s` is `null`.

## Keys

Top level: every `RuleSpec` field (`ruleset`, `period_s`, `n_periods`, `ot_s`, `halves`,
`running_clock`, `timeouts`, `possession` (`arrow` | `jump`), `bonus`, `foul_out`,
`foul_out_counts_technicals`, `tech_ft`, `shot_clock_s`, `shot_clock_reset_s`,
`clock_stops_after_fg`, `backcourt_s`, `intervals`).

| Field | Keys |
|---|---|
| `running_clock` | `stop_last_s` (the clock stops normally only in the last N s of each `stop_scope`), `stop_scope` (`period` \| `half` \| `game`), `mercy_margin` (running clock once the lead reaches N), `mercy_from_period`, `resume_below_margin` |
| `timeouts` | `scope` (`game` \| `half`), `count`, `length_s`, `short_count`, `short_length_s`, `mandatory_length_s`, `first_half`, `second_half`, `max_in_q4`, `max_late_q4`, `late_q4_s`, `per_ot`, `per_ot_length_s`, `carry_over`, `carry_to_second_half_max`, `media` (same keys) |
| `bonus` | `scope` (`period` \| `half`), `one_and_one_from`, `two_shots_from`, `late_period_s`, `late_two_shots_from`, `ot_two_shots_from`, `ot_counts_toward_last_period`. Thresholds name the team foul that first gives free throws |
| `tech_ft` | `n_ft`, `shooter` (`any` \| `fouled`), `resume` (`poi` \| `division_line` \| `frontcourt` \| `sideline` \| `spot`), `flagrant_n_ft`, `flagrant_shooter`, `flagrant_resume`, `flagrant_disqualifies` |
| `clock_stops_after_fg` | `q1_3_last_s` (each regulation period except the last), `q4_ot_last_s` (the last regulation period and every OT) |
| `intervals` | `between_periods_s`, `halftime_s`, `before_ot_s` |

## Example: an AAU tournament (fill in from the event's own rules sheet)

The values below show the format only. They are not sourced.

```yaml
ruleset: aau_example
period_s: 960            # 2 x 16 min halves
n_periods: 2
halves: true
running_clock:
  stop_last_s: 120       # the clock stops normally only in the last 2:00 of the second half
  stop_scope: game
  mercy_margin: 20
timeouts:
  count: 2               # merged into the nfhs timeouts; the other keys are kept
bonus:
  scope: half
  two_shots_from: 10
```

`no_shot_clock.yaml` in this folder is a working override for states without a shot clock.
