"""Pipeline numbers of several runs of each README game (research note §10.14's table), and made or missed paired
against the first tag (pair_shots.py). usage: pipe_summ.py TAG [TAG ...]"""
import json, re, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = Path(__file__).parent
for game in ("2024-12-25-LAL-GSW", "2017-06-12-CLE-GSW"):
    print("==", game)
    for tag in sys.argv[1:]:
        out = ROOT / "runs/nba" / game / "runs" / tag
        if not (out / "timeline.json").is_file():
            print(f"  {tag}: no timeline"); continue
        t = json.loads((out / "timeline.json").read_text())
        folds = [f.get("timeline", f).get("accuracy") if isinstance(f, dict) else None for f in t["folds"]]
        md = (out / "report_timeline.md").read_text() if (out / "report_timeline.md").is_file() else ""
        m = re.search(r"Box-score error[^*]*\*\*([0-9.]+)\*\*", md)
        s, tl, en = t["shots"], t["timeline"], t["engine"]
        print(f"  {tag}: timeline {tl['accuracy']:.4f} (folds {folds}) engine {en['accuracy']:.4f} changes P/R {tl['precision']:.2f}/{tl['recall']:.2f} | "
              f"candidates {s['candidates']} real {s['real_accuracy']:.4f} ft {s['ft_accuracy']:.4f} made {s['made_accuracy']:.4f} kept {s['kept']} | "
              f"box-score {m.group(1) if m else '?'} | switch {t['attack']['switch_t']:.0f}s")
    for tag in sys.argv[2:]:
        r = subprocess.run([str(ROOT / ".venv/bin/python"), str(P / "pair_shots.py"), str(ROOT / "runs/nba" / game), sys.argv[1], tag],
                           capture_output=True, text=True, cwd=ROOT)
        lines = [x for x in r.stdout.splitlines() if "made_ok" in x or "in both" in x] or [r.stderr.strip()[-300:]]
        for x in lines:
            print("  pair", sys.argv[1], "vs", tag, ":", x.strip())
