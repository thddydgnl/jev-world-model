"""API key resolution: env var -> .env -> macOS Keychain.

Contract: everything downstream reads TYPESAFE_API_KEY (설계.md §8).
The key is never logged, never written to artifacts, never sent anywhere
but api.typesafe.ai.
"""
import os
import subprocess
from pathlib import Path

_SERVICE = "TYPESAFE_API_KEY"
_ROOT = Path(__file__).resolve().parent.parent


def _from_dotenv() -> str | None:
    path = _ROOT / ".env"
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == _SERVICE:
            return value.strip().strip("'\"") or None
    return None


def _from_keychain() -> str | None:
    try:
        out = subprocess.run(
            ["security", "find-generic-password",
             "-a", os.environ.get("USER", ""), "-s", _SERVICE, "-w"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def get_api_key() -> str:
    for source in (lambda: os.environ.get(_SERVICE), _from_dotenv, _from_keychain):
        key = source()
        if key:
            return key
    raise RuntimeError(
        f"{_SERVICE} not found. Set the env var, add it to .env, "
        f"or store it in Keychain (see README)."
    )
