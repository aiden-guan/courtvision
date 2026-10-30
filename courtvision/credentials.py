"""Local Jev key setup; never evaluate .env contents as shell code."""
from __future__ import annotations

import getpass
import json
import os
from pathlib import Path
import re
import shlex
import tempfile
from typing import Mapping


def runtime_env(path: Path | None = None, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Read the local Jev key on each new studio session. Exported keys take precedence."""
    env = dict(os.environ if base is None else base)
    if env.get("TYPESAFE_API_KEY", "").strip():
        return env
    path = path or Path.cwd() / ".env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return env
    for line in lines:
        match = re.match(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        try:
            parts = shlex.split(match[1], comments=True)
        except ValueError:
            continue
        if len(parts) == 1 and parts[0] and not parts[0].startswith("your_"):
            env["TYPESAFE_API_KEY"] = parts[0]
    return env


def save_jev_key(key: str, path: Path) -> None:
    key = key.strip()
    if not key or any(char.isspace() for char in key) or any(ord(char) < 32 for char in key):
        raise ValueError("Paste the API key only, without an export command or whitespace.")
    if path.is_symlink():
        raise ValueError("The .env file is a symbolic link. Choose a regular local .env file.")
    original = path.read_text() if path.exists() else ""
    lines = [line for line in original.splitlines()
             if not re.match(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=", line)]
    lines.append("TYPESAFE_API_KEY=" + json.dumps(key))
    # The temporary file is private from creation; replace atomically, preserving other settings.
    fd, temporary = tempfile.mkstemp(prefix=".env-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write("\n".join(lines) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    print("\nCourtVision / Jev setup\n")
    print("Paste your TypeSafe API key below and press Return.")
    print("Input is hidden. The key stays in this project's ignored .env file (owner access only).")
    try:
        key = getpass.getpass("\nTypeSafe API key: ")
        save_jev_key(key, Path.cwd() / ".env")
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled; no key saved.")
        return
    except (ValueError, OSError) as error:
        print(f"\nCould not save key: {error}")
        return
    print("\nKey saved. Open or restart a video/demo to enable Jev for that session.")
    print("The studio picks up the key automatically. No server restart needed.")
    print("Key validity is checked by TypeSafe when the session makes its first decision.\n")


if __name__ == "__main__":
    main()
