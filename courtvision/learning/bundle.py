"""Model bundles: everything ``courtvision train`` learns, applied to a run with ``pipeline.model_bundle``.

A bundle is a directory::

    bundle.json        engine overrides (EventEngineConfig field -> value), metrics, provenance, rights
    cue_models.json    events.calibration CueModelSet (optional)
    shot_model.json    stats.shot_quality ShotQualityModel (optional)
    glyph_bank.json    perception.scorebug GlyphBank (optional)

``bundle.json["rights"]`` is "shippable" only when every training game was ``own`` / ``licensed`` with
``training_allowed`` (TRAINING.md §1.5); otherwise "research_only", and ``promote`` keeps it out of the
shipping channel. ``bundle.json["id"]`` is the sha256 of the bundle's contents (the version promoted).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ModelBundle", "apply_bundle"]

_FILES = ("cue_models.json", "shot_model.json", "glyph_bank.json")


@dataclass
class ModelBundle:
    path: Path
    engine_overrides: dict[str, Any] = field(default_factory=dict)
    rights: str = "research_only"
    metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    @classmethod
    def load(cls, path: str | Path) -> ModelBundle:
        p = Path(path)
        meta = json.loads((p / "bundle.json").read_text())
        return cls(p, dict(meta.get("engine_overrides", {})), str(meta.get("rights", "research_only")),
                   dict(meta.get("metrics", {})), dict(meta.get("provenance", {})), str(meta.get("id", "")))

    def file(self, name: str) -> Path | None:
        f = self.path / name
        return f if f.is_file() else None

    def save(self) -> ModelBundle:
        self.path.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(json.dumps(self.engine_overrides, sort_keys=True, default=str).encode())
        for name in _FILES:
            f = self.file(name)
            if f is not None:
                h.update(name.encode() + f.read_bytes())
        self.id = h.hexdigest()[:16]
        meta = {"id": self.id, "engine_overrides": self.engine_overrides, "rights": self.rights,
                "metrics": self.metrics, "provenance": self.provenance}
        (self.path / "bundle.json").write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
        return self


def apply_bundle(cfg: Any, bundle: ModelBundle | str | Path) -> Any:
    """``AppConfig`` with the bundle's engine overrides and model files (unknown engine fields raise)."""
    b = bundle if isinstance(bundle, ModelBundle) else ModelBundle.load(bundle)
    known = {f.name for f in dataclasses.fields(cfg.events)}
    unknown = sorted(set(b.engine_overrides) - known)
    if unknown:
        raise ValueError(f"bundle {b.path}: unknown engine settings {unknown}")
    events = dataclasses.replace(cfg.events, **b.engine_overrides)
    cue = b.file("cue_models.json")
    if cue is not None:
        events = dataclasses.replace(events, cue_model_path=str(cue))
    cfg = dataclasses.replace(cfg, events=events)
    shot = b.file("shot_model.json")
    if shot is not None:
        cfg = dataclasses.replace(cfg, shot_model=dataclasses.replace(cfg.shot_model, path=str(shot)))
    glyphs = b.file("glyph_bank.json")
    if glyphs is not None:
        cfg = dataclasses.replace(cfg, perception=dataclasses.replace(cfg.perception, glyph_bank_path=str(glyphs)))
    return cfg
