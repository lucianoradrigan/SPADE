"""configs/assistant.yaml: runtime, pinned model, determinism and limits (task F1.2)."""

from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlparse

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "assistant.yaml"
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
REQUIRED_LIMITS = ("max_tool_calls_per_turn", "max_experiments_per_launch", "max_total_sim_seconds")


class ConfigError(ValueError):
    pass


class RemoteRuntimeRefused(ConfigError):
    """base_url points outside this machine and allow_remote_runtime is not set."""


def is_loopback(base_url: str) -> bool:
    host = urlparse(base_url).hostname or ""
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_runtime_url(base_url: str, allow_remote_runtime: bool) -> list[str]:
    """Raise when the runtime is remote and not explicitly allowed; return warnings otherwise."""
    if urlparse(base_url).scheme not in ("http", "https"):
        raise ConfigError(f"base_url must be http(s)://..., got {base_url!r}")
    if is_loopback(base_url):
        return []
    if not allow_remote_runtime:
        raise RemoteRuntimeRefused(
            f"base_url {base_url!r} is not on this machine: the conversation and every tool result would leave it. "
            "Use 127.0.0.1 / ::1 / localhost, or set allow_remote_runtime: true in configs/assistant.yaml on purpose.")
    return [f"remote runtime {base_url}: conversations leave this machine (allow_remote_runtime: true)"]


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> dict:
    with Path(path or CONFIG_PATH).open() as f:
        cfg = yaml.safe_load(f)
    cfg = {**cfg, **(overrides or {})}
    if cfg.get("runtime") != "ollama":
        raise ConfigError(f"runtime must be 'ollama' (decision B1), got {cfg.get('runtime')!r}")
    for key in ("base_url", "model", "temperature", "seed", "limits", "timeout_s"):
        if key not in cfg:
            raise ConfigError(f"configs/assistant.yaml: missing {key!r}")
    missing = [k for k in REQUIRED_LIMITS if k not in cfg["limits"]]
    if missing:
        raise ConfigError(f"configs/assistant.yaml limits: missing {missing}")
    cfg["warnings"] = check_runtime_url(cfg["base_url"], bool(cfg.get("allow_remote_runtime", False)))
    return cfg


AVATAR_FILES = ("base.webp", "hand.webp")
REPO_ROOT = CONFIG_PATH.parents[1]


def avatar_assets(cfg: dict | None = None) -> dict:
    """The avatar layers named by ``persona.avatar_dir`` (relative to the repository root, or absolute).
    Returns {"ok", "dir", "files", "missing", "name", "tagline"}; when a layer is missing ``ok`` is False,
    the UI shows a neutral silhouette and the reason is logged. Never raises (FE-2, point 3)."""
    import logging

    cfg = cfg if cfg is not None else load_config()
    persona = cfg.get("persona") or {}
    raw = persona.get("avatar_dir") or ""
    d = Path(raw) if Path(raw).is_absolute() else REPO_ROOT / raw
    files = {f: d / f for f in AVATAR_FILES}
    missing = [f for f, p in files.items() if not p.is_file()]
    if missing:
        logging.getLogger(__name__).warning("assistant avatar: %s missing in %s; using a neutral silhouette", ", ".join(missing), d)
    return {"ok": not missing and bool(raw), "dir": d, "files": files, "missing": missing,
            "name": persona.get("name", "SPADE assistant"), "tagline": persona.get("tagline", "AI assistant · local model")}
