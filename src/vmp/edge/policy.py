"""Edge runtime policy: what the device may do off the network and with audio.

`enforce(policy, action)` is the single check point used by the edge runtime
before any network call, cloud fallback or audio write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FALLBACKS = ("cloud", "degrade", "refuse")
RETENTION_AUDIO = ("none", "local")

# Action kinds the runtime asks about.
ACTION_NETWORK = "network"
ACTION_CLOUD_FALLBACK = "cloud_fallback"
ACTION_STORE_AUDIO = "store_audio"


class PolicyViolation(RuntimeError):
    """Raised by `enforce` when the policy refuses an action."""


@dataclass(frozen=True)
class Action:
    kind: str
    target: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class EdgePolicy:
    offline_only: bool = True
    allowed_egress: list[str] = field(default_factory=list)
    max_ttfa_ms: int = 3000
    fallback: str = "degrade"
    retention: dict[str, str] = field(default_factory=lambda: {"audio": "none"})

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.fallback not in FALLBACKS:
            problems.append(f"fallback {self.fallback!r} not in {list(FALLBACKS)}")
        if self.max_ttfa_ms <= 0:
            problems.append("max_ttfa_ms must be positive")
        audio = self.retention.get("audio", "none")
        if audio not in RETENTION_AUDIO:
            problems.append(f"retention.audio {audio!r} not in {list(RETENTION_AUDIO)}")
        if self.offline_only and self.allowed_egress:
            problems.append("offline_only with non-empty allowed_egress is contradictory")
        if self.offline_only and self.fallback == "cloud":
            problems.append("offline_only with fallback='cloud' is contradictory")
        for host in self.allowed_egress:
            if not host or "://" in host:
                problems.append(f"allowed_egress entry must be a host, got {host!r}")
        return problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgePolicy:
        return cls(
            offline_only=bool(data.get("offline_only", True)),
            allowed_egress=[str(h) for h in data.get("allowed_egress", [])],
            max_ttfa_ms=int(data.get("max_ttfa_ms", 3000)),
            fallback=str(data.get("fallback", "degrade")),
            retention={str(k): str(v) for k, v in data.get("retention", {"audio": "none"}).items()},
        )


def _host_allowed(host: str | None, allowed: list[str]) -> bool:
    if host is None:
        return False
    for pattern in allowed:
        if pattern == "*" or host == pattern:
            return True
        if pattern.startswith("*.") and host.endswith(pattern[1:]):
            return True
    return False


def check(policy: EdgePolicy, action: Action | dict[str, Any]) -> tuple[bool, str]:
    """Return `(allowed, reason)` without raising."""
    a = action if isinstance(action, Action) else Action(**action)
    if a.kind == ACTION_NETWORK:
        if policy.offline_only:
            return False, f"offline_only: network call to {a.target!r} refused"
        if not _host_allowed(a.target, policy.allowed_egress):
            return False, f"egress to {a.target!r} not in allowed_egress {policy.allowed_egress}"
        return True, f"egress to {a.target!r} allowed"
    if a.kind == ACTION_CLOUD_FALLBACK:
        if policy.offline_only:
            return False, "offline_only: cloud fallback refused"
        if policy.fallback != "cloud":
            return False, f"fallback policy is {policy.fallback!r}, not 'cloud'"
        return True, "cloud fallback allowed"
    if a.kind == ACTION_STORE_AUDIO:
        mode = policy.retention.get("audio", "none")
        if mode == "none":
            return False, "retention.audio is 'none': audio must not be stored"
        return True, f"audio retention {mode!r} allows local storage"
    return False, f"unknown action kind {a.kind!r}"


def enforce(policy: EdgePolicy, action: Action | dict[str, Any]) -> str:
    """Raise `PolicyViolation` when refused; return the reason string when allowed."""
    allowed, reason = check(policy, action)
    if not allowed:
        raise PolicyViolation(reason)
    return reason


def policy_from_config(data: dict[str, Any]) -> EdgePolicy:
    """Build from the `[policy]` table of `configs/edge.toml`."""
    return EdgePolicy.from_dict(data.get("policy", data))


__all__ = [
    "ACTION_CLOUD_FALLBACK",
    "ACTION_NETWORK",
    "ACTION_STORE_AUDIO",
    "FALLBACKS",
    "RETENTION_AUDIO",
    "Action",
    "EdgePolicy",
    "PolicyViolation",
    "check",
    "enforce",
    "policy_from_config",
]
