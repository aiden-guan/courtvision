import json, sys, math
from itertools import combinations
def mcnemar(a, b):
    oa = sum(1 for x, y in zip(a, b) if x and not y); ob = sum(1 for x, y in zip(a, b) if y and not x)
    n = oa + ob; k = min(oa, ob)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n) if n else 1.0
    return oa, ob, p
for path in sys.argv[1:]:
    d = json.load(open(path)); ms = d["models"]
    print("==", path, "linked" if d.get("linked") else "top")
    for m, o in ms.items():
        print(f"  {m.split('/')[-1]:30s} n={o['n']} hit={o['hit_rate']:.4f} wrong={o['wrong_rate']:.4f} miss={o['miss_rate']:.4f} "
              f"med={o['median_err_px_1080']:.2f} people={o.get('person_recall_vs_teacher') or float('nan'):.4f}")
    for a, b in combinations(ms, 2):
        oa, ob, p = mcnemar(ms[a]["hits"], ms[b]["hits"])
        print(f"  {a.split('/')[-1]} vs {b.split('/')[-1]}: only first {oa}, only second {ob}, p={p:.3g}")
