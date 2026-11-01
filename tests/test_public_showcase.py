"""Keep the public deployment independent of the local inference application."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "showcase"


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.elements = []
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_public_deployment_contains_only_static_recording_assets():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert config["framework"] is None
    assert config["buildCommand"] == ""
    assert config["installCommand"] == ""
    assert config["outputDirectory"] == "showcase"
    assert not ({"functions", "builds", "rewrites", "crons", "services"} & config.keys())
    allowed = {".html", ".css", ".woff2", ".txt", ".mp4", ".webp"}
    assert all(p.suffix in allowed for p in SITE.rglob("*") if p.is_file())
    assert len(list(SITE.rglob("*.mp4"))) == 1


def test_recording_loads_only_when_requested_and_all_assets_resolve():
    page = Page((SITE / "index.html").read_text())
    assert not any(tag in {"script", "iframe", "input", "form"} for tag, _ in page.elements)
    videos = [attrs for tag, attrs in page.elements if tag == "video"]
    assert len(videos) == 1
    video = videos[0]
    assert video["preload"] == "none"
    assert "controls" in video and "playsinline" in video
    assert "autoplay" not in video
    for tag, attrs in page.elements:
        for key in ("href", "src", "poster"):
            path = attrs.get(key, "")
            if path and not urlsplit(path).scheme and not path.startswith(("#", "/")):
                assert (SITE / path).is_file(), (tag, path)
    css = (SITE / "styles.css").read_text()
    for path in re.findall(r'url\("([^"]+)"\)', css):
        assert (SITE / path).is_file()
    meta = json.loads((ROOT / "docs/showcase-recording.json").read_text())
    recording = ROOT / meta["video"]
    assert hashlib.sha256(recording.read_bytes()).hexdigest() == meta["video_sha256"]
    assert meta["video_sha256"][:12] in recording.name
    assert 29 <= meta["duration_s"] <= 30.1
    assert meta["frames"] > 850
    assert (SITE / video["poster"]).resolve() == (ROOT / meta["poster"]).resolve()
    sources = [attrs["src"] for tag, attrs in page.elements if tag == "source"]
    assert sources == ["media/" + recording.name]
