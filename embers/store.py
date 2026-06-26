"""Durable control-plane state — a SQLite registry so the control plane recovers
its placement across a restart.

Persists two things:
  * assignments  — model → node + the model's spec (the placement registry)
  * snapshots    — model → node that holds a warm GPU snapshot (the snapshot index,
    so a recovered control plane knows which models can fast-restore where)

SQLite is the in-repo stand-in for a Postgres registry: same SQL, swappable later.
Full HA (multi-replica consensus) is out of scope — that needs etcd/Raft; this gives
DURABILITY (survives a process restart), the realistic single-control-plane guarantee.
"""
from __future__ import annotations

import json
import sqlite3
import threading

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assignments (
    model    TEXT PRIMARY KEY,
    node_id  TEXT NOT NULL,
    spec     TEXT NOT NULL          -- JSON: vram_mb, tensor_parallel_size, adapters, ...
);
CREATE TABLE IF NOT EXISTS snapshots (
    model    TEXT PRIMARY KEY,
    node_id  TEXT NOT NULL
);
"""


class ControlPlaneStore:
    """Durable registry. `path=":memory:"` for tests; a file path for production."""

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # --- assignments (the placement registry) -----------------------------

    def save_assignment(self, model: str, node_id: str, spec: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO assignments(model, node_id, spec) VALUES (?,?,?)",
                (model, node_id, json.dumps(spec)))
            self._db.commit()

    def remove_assignment(self, model: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM assignments WHERE model=?", (model,))
            self._db.commit()

    def load_assignments(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT model, node_id, spec FROM assignments").fetchall()
        return [{"model": m, "node_id": n, "spec": json.loads(s)} for m, n, s in rows]

    # --- snapshot index (which node holds a warm model) -------------------

    def set_snapshot(self, model: str, node_id: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO snapshots(model, node_id) VALUES (?,?)",
                (model, node_id))
            self._db.commit()

    def snapshot_index(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT model, node_id FROM snapshots").fetchall()
        return {m: n for m, n in rows}

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM assignments")
            self._db.execute("DELETE FROM snapshots")
            self._db.commit()

    def close(self) -> None:
        self._db.close()
