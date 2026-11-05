# Round 8 on the owner's Mac (research note §13)

The scripts that made and scored round 8's candidates on the Mac, kept so that the results can be reproduced. They are
research tools with fixed paths under `runs/` and `data/` (git-ignored) and the README's two games, which only the
Mac has; they run from the repository root.

During round 8 the Mac's CPU was shared with much other work (load averages of 20-60 on 11 cores, 18 GB of RAM with
16-24 GB of swap), and ONNX Runtime on the CPU took 13-19 s a frame. So the networks run on the GPU (PyTorch MPS,
fp32): `mps_student.MpsNet` loads a checkpoint (`student_from_onnx`'s exact copy of an export, or a run's `last.pt`).
On three frames v2 on MPS and v2's ONNX export gave the same balls and boxes to 6.4e-5.

| Script | What it is |
|---|---|
| `mps_student.py` | the network on MPS; `patch_bench()` makes `detector_bench`'s `.pt` models run there |
| `prep_mps.py` | `student_finetune.py prep-video`, line for line, with the network on MPS in batches of 8 |
| `prep_check.py` | checks `prep_mps.py` against the tool itself (same rows, balls, labels, meta and stored frames on two clips) and MPS against ONNX |
| `bench_mps.py` | `tools/detector_bench.py` with its `.pt` models on MPS |
| `mi_mps.py` | `student_finetune.py mark-ignore` with the labeller on MPS |
| `cp_parallel.py` | `student_finetune.py court-persons` with the per-clip court fits in parallel processes and the labeller on MPS |
| `people_split.py`, `people_paired.py` | the teacher's people on the two games split by where they stand (§10.2), and the paired players on the court (per box and per clip) |
| `summ.py` | the ball tables of `detector_bench --ball-folder` runs, paired (McNemar) |
| `eval.sh` | every held-out test of §10.8 and §13.2 for a list of checkpoints, against v2 |
| `chain.sh` | the round: the new clips' ball labels, R8a, E-BARD, R8e, SportsMOT's officials promoted, R8f, the scores, court-persons, R8b |
| `pipeline.sh`, `pipe_summ.py`, `pair_shots.py` | the pipeline on both games for a checkpoint (or for v2's detections with another referee check), its numbers, and made-or-missed paired shot by shot |

The footage, labels and models these read are research-only (docs/TRAINING.md §1).
