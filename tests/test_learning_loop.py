"""The learning loop on simulated games: ingest (pseudo play-by-play) -> train -> promote, and its gates."""

from __future__ import annotations

import json
import shutil

import pytest

from courtvision.cli import main, simulate_config
from courtvision.eval.events import match_events
from courtvision.learning.bundle import ModelBundle, apply_bundle
from courtvision.learning.dataset import PBP_KINDS, GameDataset, frame_from_dict, frame_to_dict, load_datasets
from courtvision.learning.ingest import IngestConfig, ingest_game, sim_pbp
from courtvision.learning.promote import PromoteConfig, evaluate_bundle, promote
from courtvision.learning.train import TrainConfig, train
from courtvision.pipeline import make_source
from courtvision.sim.simulator import GameSimulator

SECONDS = 150.0
# 150 s games have few made field goals: the gate takes 2 (seeds 2 and 6 pass it; with fewer than 5 the scorer's
# lag is not estimated and seeds 1, 3, 5 fail it)
ALIGN = {"accept_min_made": 2}


def _ingest(root, seed: int, **kw):
    cfg = simulate_config("mono", seed=seed, seconds=SECONDS)
    r = GameSimulator(cfg.sim).run()
    src = make_source(cfg, sim_result=r, realtime=False)
    icfg = IngestConfig(rights_status=kw.pop("rights", "own"), training_allowed=kw.pop("allowed", True), align=ALIGN, **kw)
    ds = ingest_game(cfg, src, sim_pbp(r.events, r.frames), root / f"g{seed}", f"g{seed}", icfg)
    return ds, r


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets")
    train_sets = [_ingest(root / "train", s) for s in (2, 6)]
    gold = _ingest(root / "gold", 101, gold=True)
    return root, train_sets, gold


def test_ingest_writes_a_labelled_game(games) -> None:
    _, train_sets, (gold, sim) = games
    ds = train_sets[0][0]
    m = ds.manifest
    assert m.frames == int(SECONDS * 30) and m.source == "sim" and not ds.verify()
    assert {"tracks.jsonl.gz", "labels.jsonl", "engine_events.jsonl", "clips.jsonl", "cue_rows.jsonl", "shots.csv",
            "scoreboard.jsonl", "roster.json"} <= set(m.files)
    f = next(ds.frames())
    assert frame_from_dict(frame_to_dict(f)) == f  # the cached tracks round-trip
    # the labels are the game's real events: matched to the simulator's truth (they come from a PBP of it)
    rep = match_events(gold.labels(), sim.events, kinds=list(PBP_KINDS), tol_s=1.5, match_player=False)
    for k in ("shot_attempt", "shot_missed", "rebound"):
        if rep.per_kind[k].n_truth:
            assert rep.per_kind[k].f1 >= 0.8, (k, rep.per_kind[k].f1)


def test_train_tunes_promote_gates(games, tmp_path) -> None:
    root, train_sets, (gold, _) = games
    datasets = [ds for ds, _ in train_sets]
    # a deployed bundle with a crippling setting: the image holder needs an implausibly high score
    bad = ModelBundle(tmp_path / "bad", {"holder_min_score": 0.97}, "shippable").save()
    models = tmp_path / "models"
    shutil.copytree(bad.path, models / "current")
    search = {"holder_min_score": [0.1, 0.97]}
    cand = train(datasets, tmp_path / "cand", TrainConfig(search=search, workers=1, fit_shots=False), base_bundle=bad)
    assert cand.engine_overrides == {"holder_min_score": 0.1}
    assert cand.metrics["engine"]["macro_f1"] > cand.metrics["engine_start"]["macro_f1"]
    assert cand.rights == "shippable" and [g["game_id"] for g in cand.provenance["games"]] == ["g2", "g6"]
    assert (tmp_path / "cand" / "cue_models.json").is_file()
    # gold sets cannot train
    with pytest.raises(ValueError, match="gold"):
        train([gold], tmp_path / "x", TrainConfig(search=search, workers=1))
    # the candidate beats the deployed bundle on the gold game: promoted, the old one kept in history
    d = promote(cand, models, [gold], PromoteConfig(channel="ship"))
    assert d["promoted"], d["reasons"]
    assert ModelBundle.load(models / "current").id == cand.id and (models / "history" / bad.id).is_dir()
    # the same candidate again: no gain, not promoted; every decision is logged
    d2 = promote(cand, models, [gold])
    assert not d2["promoted"] and "does not beat" in d2["reasons"][0]
    log = [json.loads(line) for line in (models / "promotions.jsonl").read_text().splitlines()]
    assert [x["promoted"] for x in log] == [True, False]
    # a bundle applies to a run's configuration
    cfg = apply_bundle(simulate_config("mono", seed=2, seconds=10.0), models / "current")
    assert cfg.events.holder_min_score == 0.1 and cfg.events.cue_model_path.endswith("cue_models.json")
    assert evaluate_bundle(ModelBundle.load(models / "current"), [gold])["macro_f1"] == d["candidate_score"]["macro_f1"]


def test_rights_and_frozen_gold_gates(games, tmp_path) -> None:
    root, train_sets, (gold, _) = games
    research, _ = _ingest(tmp_path / "rnd", 2, rights="research_only", allowed=False)
    assert not research.manifest.shippable
    cand = train([train_sets[0][0], research], tmp_path / "cand", TrainConfig(search={}, workers=1, fit_cues=False,
                                                                                 fit_shots=False))
    assert cand.rights == "research_only"  # one research-only game keeps the bundle out of shipping
    d = promote(cand, tmp_path / "models", [gold], PromoteConfig(channel="ship", min_gain=-1.0), dry_run=True)
    assert not d["promoted"] and any("rights" in r for r in d["reasons"])
    # a gold set changed after ingest is not frozen
    g2 = tmp_path / "gold2"
    shutil.copytree(gold.path, g2)
    with (g2 / "labels.jsonl").open("a") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="frozen"):
        promote(cand, tmp_path / "models", [GameDataset(g2)], dry_run=True)
    with pytest.raises(ValueError, match="not gold"):
        promote(cand, tmp_path / "models", [train_sets[0][0]], dry_run=True)


def test_cli_loop(tmp_path) -> None:
    out = tmp_path / "ds"
    assert main(["-q", "ingest", "--sim-seed", "4", "--mode", "truth3d", "--seconds", "60", "--out", str(out),
                 "--set", "sim.period_s=600"]) == 0
    ds = load_datasets([out])
    assert len(ds) == 1 and ds[0].manifest.rights_status == "own" and ds[0].manifest.training_allowed
    # 60 s has too few makes for the gate; mark it accepted for the CLI path
    (ds[0].path / "manifest.json").write_text(json.dumps({**json.loads((ds[0].path / "manifest.json").read_text()),
                                                          "accepted": True}))
    assert main(["-q", "train", str(out), "--out", str(tmp_path / "b"), "--params", "steal_vote_frac",
                 "--workers", "1", "--no-shots"]) == 0
    assert ModelBundle.load(tmp_path / "b").rights == "shippable"
    assert main(["-q", "train", str(out), "--out", str(tmp_path / "c"), "--params", "nope"]) == 2
