"""Per-stage run health.

`runs.status` says whether the PROCESS finished ('ok' | 'error'). It says nothing
about whether the data was whole: runs #99-#113 (Aug 24 - Sep 2, 2026) were all
`status='ok'` while IBKR was offline, so every one of them was portfolio-blind and
nothing on the dashboard said so for five weeks.

`health` answers the other question - "can I trust this output?" - per stage:

    ok        stage ran on complete inputs
    degraded  stage ran but on partial/fallback inputs (Gateway offline, Ollama down,
              Yahoo throttled, research budget exhausted ...) - output exists, treat
              with care; the note says exactly what was missing
    failed    stage produced nothing usable

The run's overall health is the worst stage. Persisted as `runs.health` +
`runs.health_json` (and the same on `portfolio_runs`); the dashboard shows a badge
and the per-stage notes, the launcher's banner reads the same field.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

OK = "ok"
DEGRADED = "degraded"
FAILED = "failed"
_RANK = {OK: 0, DEGRADED: 1, FAILED: 2}

# Stage names are fixed so the dashboard/doctor can render them in a stable order.
STAGES = ("fundamentals", "technicals", "regime", "sentiment", "broker",
          "research", "triggers", "risk", "synthesis")

GLYPH = {OK: "🟢", DEGRADED: "🟠", FAILED: "🔴"}


@dataclass
class StageHealth:
    status: str = OK
    note: str = ""

    def worsen(self, status: str, note: str) -> None:
        """Only ever move toward worse; keep the first (most specific) note per level."""
        if _RANK[status] > _RANK[self.status]:
            self.status, self.note = status, note
        elif _RANK[status] == _RANK[self.status] and note and not self.note:
            self.note = note


@dataclass
class RunHealth:
    stages: dict[str, StageHealth] = field(default_factory=dict)

    # -- recording ---------------------------------------------------------
    def stage(self, name: str) -> StageHealth:
        return self.stages.setdefault(name, StageHealth())

    def ok(self, name: str, note: str = "") -> None:
        st = self.stage(name)
        if not st.note and note:
            st.note = note

    def degraded(self, name: str, note: str) -> None:
        self.stage(name).worsen(DEGRADED, note)

    def failed(self, name: str, note: str) -> None:
        self.stage(name).worsen(FAILED, note)

    # -- reading -----------------------------------------------------------
    @property
    def overall(self) -> str:
        if not self.stages:
            return OK
        return max((s.status for s in self.stages.values()), key=_RANK.__getitem__)

    def problems(self) -> list[tuple[str, str, str]]:
        """[(stage, status, note)] for every non-ok stage, worst first."""
        out = [(n, s.status, s.note) for n, s in self.stages.items() if s.status != OK]
        return sorted(out, key=lambda t: -_RANK[t[1]])

    def summary(self) -> str:
        """One line for logs and the launcher banner."""
        probs = self.problems()
        if not probs:
            return "health ok"
        return f"health {self.overall}: " + "; ".join(
            f"{n} {st}" + (f" ({note})" if note else "") for n, st, note in probs)

    # -- persistence -------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps({n: asdict(s) for n, s in self.stages.items()})

    @classmethod
    def from_json(cls, text: str | None) -> "RunHealth":
        rh = cls()
        if not text:
            return rh
        try:
            data: dict[str, Any] = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return rh
        for name, d in data.items():
            if isinstance(d, dict):
                rh.stages[name] = StageHealth(status=str(d.get("status", OK)),
                                              note=str(d.get("note", "")))
        return rh


def badge(health: str | None) -> str:
    """Glyph + word for dashboard/launcher use; None (pre-feature runs) -> 'unknown'."""
    if health not in _RANK:
        return "⚪ unknown"
    return f"{GLYPH[health]} {health}"
