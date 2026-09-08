"""Score today's universe with the latest walk-forward fold.

Used by the pipeline when a trained model exists. The pillar is only WEIGHTED when
config `scoring.pillars.learned` > 0 AND the model's verdict passed (or
`scoring.learned.force: true`); otherwise scores are still computed and persisted
so the dashboard can show the dark pillar next to the live ones.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config
from . import features as F


def models_root() -> Path:
    from ..data.history_store import HistoryStore
    return HistoryStore().root / "models" / "rank_v1"


def latest_meta() -> dict[str, Any] | None:
    p = models_root() / "latest.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def usable() -> tuple[bool, str]:
    """(can the pipeline weight this pillar?, reason)."""
    meta = latest_meta()
    if not meta:
        return False, "no trained model"
    if meta.get("feature_version") != F.FEATURE_VERSION:
        return False, f"model feature_version {meta.get('feature_version')} != {F.FEATURE_VERSION}"
    if not (meta.get("verdict") or {}).get("pass") and not config.get("scoring.learned.force", False):
        return False, "kill criteria not passed (scoring.learned.force to override)"
    return True, "ok"


def score_today(symbols: list[str], subsectors: dict[str, str],
                as_of: date | None = None) -> tuple[dict[str, float], dict[str, Any]]:
    """Returns ({symbol: raw model score}, info). Empty dict when no model."""
    meta = latest_meta()
    if not meta:
        return {}, {"note": "no trained model"}
    from ..data.history_store import HistoryStore
    import lightgbm as lgb
    store = HistoryStore()
    # The as-of date comes from the manifest: building a full date x symbol pivot of
    # the entire store just to read its last index was the single most expensive line
    # in the run, and it grows with the universe.
    d = pd.Timestamp(as_of) if as_of else store.last_date()
    if d is None:
        return {}, {"note": "history store has no prices"}
    frame = F.build_feature_frame(store, symbols, subsectors, [d], with_labels=False)
    if frame.empty:
        return {}, {"note": "feature frame empty for today"}
    feats = meta["features"]
    frame = F.rank_transform(frame, feats)
    X = frame.reindex(columns=feats)
    booster = lgb.Booster(model_file=meta["model_path"])
    pred = booster.predict(X)
    out = {s: float(p) for s, p in zip(frame["symbol"], pred)}
    info = {"model_date": meta.get("test_date"), "as_of": str(d.date()), "n": len(out),
            "oos_ic": (meta.get("summary") or {}).get("ic_mean"),
            "verdict_pass": (meta.get("verdict") or {}).get("pass")}
    return out, info
