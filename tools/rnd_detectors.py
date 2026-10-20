"""R&D-only detector factories for ``courtvision run|serve --detector rnd-yolo:<weights>``.

**Not shipped.** This file lives in ``tools/`` (outside the ``courtvision`` package), so it exists
only in a source checkout. It is the single bridge from the command line to the AGPL-3.0
Ultralytics adapter in ``courtvision/perception/rnd/`` (RESEARCH.md D9). Shipped code never imports
that package (``tests/test_perception_ultralytics.py`` enforces this): ``courtvision.cli`` loads
this file by path, and only after the user has asked for ``rnd-yolo`` and seen the AGPL warning. An
installed wheel therefore cannot reach the AGPL code path at all.
"""

from __future__ import annotations

from typing import Any


def build_ultralytics(weights: str, **kwargs: Any) -> Any:
    """``UltralyticsDetector(weights)`` (extra ``rnd-agpl``; imported here, lazily)."""
    from courtvision.perception.rnd.ultralytics_adapter import UltralyticsDetector

    return UltralyticsDetector(str(weights), **kwargs)
