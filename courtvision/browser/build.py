"""Package the original perception modules for Pyodide. Never includes footage or credentials."""
from pathlib import Path
import hashlib
import json
import zipfile


def build() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "__init__.py", root / "types.py"]
    for folder in ("browser", "geometry", "perception", "sim", "nba"):
        paths.extend(sorted((root / folder).rglob("*.py")))
    out = root / "server" / "static" / "browser-python.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            info = zipfile.ZipInfo("courtvision/" + str(path.relative_to(root)), (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    manifest = {"sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                "sources": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    build()
