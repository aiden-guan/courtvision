"""Pair two timeline runs' held-out shot decisions shot by shot (the play-by-play shot each candidate is labelled with).
usage: pair_shots.py <game root> <tagA> <tagB>"""
import json, math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from nba_analyze import truth  # noqa: E402
from courtvision.nba.makes import ShotModels  # noqa: E402

root, tags = Path(sys.argv[1]), sys.argv[2:4]
aligned = truth(root)[1]
shots = [(float(al.t), bool(al.action.is_free_throw), bool(al.action.made)) for al in aligned.aligned
         if al.t is not None and al.action.is_shot and not al.action.synthetic and al.action.made is not None]

def match(cands, tol=2.5):  # label_candidates' one-to-one nearest-first pairing, returning the shot index
    pairs = sorted((abs(c["t"] - t), i, j) for i, c in enumerate(cands) for j, (t, _, _) in enumerate(shots) if abs(c["t"] - t) <= tol)
    uc, us, out = set(), set(), {}
    for _, i, j in pairs:
        if i in uc or j in us: continue
        uc.add(i); us.add(j); out[i] = j
    return out

def run(tag):
    out = root / "runs" / tag
    rows = [json.loads(l) for l in (out / "shot_rows.jsonl").open()]
    tl = json.loads((out / "timeline.json").read_text())
    split = tl["attack"]["switch_t"]
    fits = {True: ShotModels.from_dict(tl["shots"]["models"]["first half"]),
            False: ShotModels.from_dict(tl["shots"]["models"]["second half"])}
    m = match(rows)
    per_shot, false_kept, n_false = {}, 0, 0
    right = {"real": 0, "ft": 0, "made": 0}; n_real = 0
    for i, r in enumerate(rows):
        f = r["features"]; mod = fits[r["t"] < split]
        real = mod.real.p(f) >= 0.5
        lab = r["label"]
        assert (lab == "false") == (i not in m), (tag, r["key"], lab)
        right["real"] += real == (lab != "false")
        if lab == "false":
            n_false += 1; false_kept += real; continue
        n_real += 1
        ft = mod.ft.p(f) >= 0.5
        made = mod.made.p({**f, "is_ft": float(ft)}) >= 0.5
        right["ft"] += ft == lab.startswith("ft"); right["made"] += made == lab.endswith("made")
        per_shot[m[i]] = {"kept": real, "ft_ok": ft == lab.startswith("ft"), "made_ok": made == lab.endswith("made")}
    print(f"{tag}: {len(rows)} candidates, {n_real} real ({len(shots)} play-by-play shots), real-or-not "
          f"{right['real'] / len(rows):.4f}, ft {right['ft'] / n_real:.4f}, made {right['made'] / n_real:.4f}; "
          f"real shots kept {sum(v['kept'] for v in per_shot.values())}, false candidates kept {false_kept} of {n_false}")
    return per_shot

def mcnemar(a, b):
    n = a + b; k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

A, B = run(tags[0]), run(tags[1])
both = sorted(set(A) & set(B))
print(f"play-by-play shots with a candidate in both: {len(both)}; only {tags[0]}: {len(set(A) - set(B))}, only {tags[1]}: {len(set(B) - set(A))}")
for key in ("made_ok", "kept", "ft_ok"):
    oa = sum(1 for j in both if A[j][key] and not B[j][key]); ob = sum(1 for j in both if B[j][key] and not A[j][key])
    print(f"  {key}: {tags[0]} {sum(A[j][key] for j in both)}, {tags[1]} {sum(B[j][key] for j in both)} of {len(both)}; "
          f"only {tags[0]} {oa}, only {tags[1]} {ob}, p {mcnemar(oa, ob):.3g}")
