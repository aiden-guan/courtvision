"""Two shot judges' calls (nba_live_judge.py --dump) paired over the shots both models' events give: the same game,
times within 1.5 s, nearest first, one to one; exact McNemar on right/wrong (a call against the truth), per game and for both ways of fitting
(each game by a judge fitted on the other, and half by half). usage: judge_pair.py A.jsonl B.jsonl"""
import json
import math
import sys
from collections import defaultdict


def mcnemar(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


rows = [defaultdict(list), defaultdict(list)]
for side, path in enumerate(sys.argv[1:3]):
    for r in map(json.loads, open(path)):
        rows[side][r["game"]].append(r)
for game in sorted(set(rows[0]) | set(rows[1])):
    a, b = rows[0][game], rows[1][game]
    pairs = sorted((abs(x["t"] - y["t"]), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if abs(x["t"] - y["t"]) <= 1.5)
    ua, ub, both = set(), set(), []
    for _, i, j in pairs:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); both.append((a[i], b[j]))
    for key in ("other_games", "halves"):  # each field is the judge's call (made?); "made" is the truth
        ok = [((x[key] == x["made"]), (y[key] == y["made"])) for x, y in both]
        ra, rb = sum(p for p, _ in ok), sum(q for _, q in ok)
        oa, ob = sum(1 for p, q in ok if p and not q), sum(1 for p, q in ok if q and not p)
        print(f"{game} {key}: {len(both)} shots in both (only A {len(a) - len(both)}, only B {len(b) - len(both)}); "
              f"A right {ra}, B right {rb}; only A {oa}, only B {ob}, p {mcnemar(oa, ob):.2g}")
