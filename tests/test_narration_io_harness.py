"""tools/latency_harness.py with injected fake clients (no network, no credentials)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tests.test_narration_io_support import FakeAnthropic, FakeTypeSafe

_PATH = Path(__file__).resolve().parents[1] / "tools" / "latency_harness.py"
_spec = importlib.util.spec_from_file_location("latency_harness", _PATH)
assert _spec is not None and _spec.loader is not None
harness = importlib.util.module_from_spec(_spec)
sys.modules["latency_harness"] = harness
_spec.loader.exec_module(harness)


def test_refuses_without_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(harness, "claude_credentials_available", lambda client=None: False)
    monkeypatch.setattr(harness, "_make_claude_client", lambda: object())
    assert harness.main(["--configs", "C1", "--n", "1", "--out", str(tmp_path)]) == 2
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert harness.main(["--configs", "J4", "--n", "1", "--out", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []  # nothing ran, nothing written


def test_unknown_config_rejected(tmp_path: Path) -> None:
    assert harness.main(["--configs", "C9", "--out", str(tmp_path)], claude_client=FakeAnthropic()) == 2


def test_sample_packet_is_real_shaped() -> None:
    facts = harness.sample_facts()
    text = json.dumps(facts)
    assert 4000 <= len(text) <= 9000  # ~1.5k tokens of JSON
    assert facts["allowed_tokens"] and not any(n in text for n in harness._HOME + harness._AWAY)  # D16
    cases = harness.jev_cases(facts)
    assert set(cases) == {"B1", "B2", "B3", "B4", "B5"}


def test_full_run_with_fakes(tmp_path: Path) -> None:
    claude = FakeAnthropic(text="{P07} has 19 points tonight.", output_tokens=400)
    jev = FakeTypeSafe()
    rc = harness.main(["--configs", ",".join(harness.ALL_CONFIGS), "--n", "3", "--pause", "0", "--network", "gym-wifi",
                       "--out", str(tmp_path)], claude_client=claude, jev_client=jev)
    assert rc == 0
    [json_path] = list(tmp_path.glob("latency_gym-wifi_*.json"))
    [md_path] = list(tmp_path.glob("latency_gym-wifi_*.md"))
    report = json.loads(json_path.read_text())
    by = {s["config"]: s for s in report["summaries"]}
    assert set(by) == set(harness.ALL_CONFIGS)
    c1 = by["C1"]
    assert c1["n"] == 3 and c1["errors"] == {} and c1["stop_reasons"] == {"end_turn": 3}
    for metric in ("t_message_start_s", "t_first_text_s", "total_s"):
        assert c1[metric]["p50"] is not None and c1[metric]["p95"] >= c1[metric]["p50"]
    assert c1["output_tokens"]["p50"] == 400 and c1["cache_read_input_tokens"]["p50"] == 880
    assert c1["output_cost_usd_per_call"]["p50"] == pytest.approx(400 * 20.0 / 1e6)
    assert c1["system_tokens"] == 870 and c1["models_seen"] == ["claude-opus-5-5"]
    # request shapes per configuration (each config: 3 streams, after one non-streaming pre-warm)
    streams = claude.stream_calls
    assert len(streams) == 12 and len(claude.create_calls) == 4
    assert all(c["max_tokens"] == 0 for c in claude.create_calls)
    c1_kw, c2_kw, c3_kw, c4_kw = streams[0], streams[3], streams[6], streams[9]
    assert c1_kw["betas"] == ["server-side-fallback-2026-07-01"] and c1_kw["fallbacks"] == "default"
    assert "Answer directly without deliberating." in c2_kw["system"][0]["text"]
    assert "Answer directly" not in c1_kw["system"][0]["text"]
    assert c3_kw["model"] == "claude-haiku-5-5" and c3_kw["thinking"] == {"type": "disabled"}
    assert "betas" not in c3_kw and "fallbacks" not in c3_kw
    assert c4_kw["model"] == "claude-haiku-5-5" and "thinking" not in c4_kw and "betas" not in c4_kw
    assert all(k["output_config"] == {"effort": "low"} and k["max_tokens"] == 1536 for k in streams)
    # Jev bundles
    j4 = by["J4"]
    assert j4["bundle"] == "B4" and j4["n"] == 3 and j4["models_seen"] == ["jev-1.13.0"]
    assert j4["wall_s"]["p95"] is not None and j4["n_questions"] == 6
    assert len(jev.calls) == 15 and all("_local" not in json.dumps(s) for s, _ in jev.calls)
    rec = report["recommended"]
    assert set(rec["claude_candidate_deadline_s"]) == {"C1", "C2", "C3", "C4"}
    assert set(rec["jev_deadlines_s"]) == {"B1", "B2", "B3", "B4", "B5"}
    md = md_path.read_text()
    assert "# Latency harness: gym-wifi" in md and "| C3 | claude-haiku-5-5 | 3 |" in md and "| J5 | B5 |" in md
    assert "message_start p50/p95" in md and "Recommended deadlines" in md


def test_refusals_are_counted(tmp_path: Path) -> None:
    claude = FakeAnthropic(stop_reason="refusal")
    rc = harness.main(["--configs", "C1", "--n", "2", "--pause", "0", "--out", str(tmp_path), "--no-prewarm"],
                      claude_client=claude)
    assert rc == 0 and claude.create_calls == []
    report = json.loads(next(tmp_path.glob("*.json")).read_text())
    s = report["summaries"][0]
    assert s["stop_reasons"] == {"refusal": 2} and s["errors"] == {"refusal": 2}
