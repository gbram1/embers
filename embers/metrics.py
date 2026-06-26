"""Phase 6 observability — metrics + a platform snapshot/dashboard.

Lightweight in-process metrics (Counter / Gauge / Histogram) rendered as
Prometheus text and JSON, plus `platform_snapshot()` which pulls a unified view
from the scheduler (GPU utilisation), autoscaler (replicas, scale events), and
loader (the cold-start win: restores vs cold_loads) — and `render_dashboard()`
to print it. The headline number this surfaces is the **snapshot hit-rate**:
restores / (restores + cold_loads).
"""
from __future__ import annotations

from typing import Any

LabelKey = tuple[tuple[str, str], ...]


def _key(labels: dict[str, str]) -> LabelKey:
    return tuple(sorted(labels.items()))


def _fmt_labels(key: LabelKey) -> str:
    if not key:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in key)
    return "{" + inner + "}"


class Counter:
    def __init__(self, name: str, help: str = ""):
        self.name, self.help = name, help
        self._v: dict[LabelKey, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        k = _key(labels)
        self._v[k] = self._v.get(k, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._v.get(_key(labels), 0.0)

    def _lines(self) -> list[str]:
        return [f"{self.name}{_fmt_labels(k)} {v}" for k, v in self._v.items()]


class Gauge:
    def __init__(self, name: str, help: str = ""):
        self.name, self.help = name, help
        self._v: dict[LabelKey, float] = {}

    def set(self, value: float, **labels: str) -> None:
        self._v[_key(labels)] = value

    def value(self, **labels: str) -> float:
        return self._v.get(_key(labels), 0.0)

    def _lines(self) -> list[str]:
        return [f"{self.name}{_fmt_labels(k)} {v}" for k, v in self._v.items()]


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


class Histogram:
    def __init__(self, name: str, help: str = ""):
        self.name, self.help = name, help
        self._obs: dict[LabelKey, list[float]] = {}

    def observe(self, value: float, **labels: str) -> None:
        self._obs.setdefault(_key(labels), []).append(value)

    def stats(self, **labels: str) -> dict[str, float]:
        xs = self._obs.get(_key(labels), [])
        if not xs:
            return {"count": 0, "sum": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        return {"count": len(xs), "sum": sum(xs),
                "p50": _pct(xs, 50), "p90": _pct(xs, 90), "p99": _pct(xs, 99)}

    def _lines(self) -> list[str]:
        out = []
        for k, xs in self._obs.items():
            base = _fmt_labels(k)[:-1] if k else "{"
            for q in (50, 90, 99):
                sep = "," if k else ""
                out.append(
                    f'{self.name}{base}{sep}quantile="{q / 100:g}"}} {_pct(xs, q)}')
            out.append(f"{self.name}_count{_fmt_labels(k)} {len(xs)}")
            out.append(f"{self.name}_sum{_fmt_labels(k)} {sum(xs)}")
        return out


class Registry:
    def __init__(self):
        self._metrics: dict[str, Any] = {}

    def counter(self, name: str, help: str = "") -> Counter:
        return self._metrics.setdefault(name, Counter(name, help))

    def gauge(self, name: str, help: str = "") -> Gauge:
        return self._metrics.setdefault(name, Gauge(name, help))

    def histogram(self, name: str, help: str = "") -> Histogram:
        return self._metrics.setdefault(name, Histogram(name, help))

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for m in self._metrics.values():
            if m.help:
                lines.append(f"# HELP {m.name} {m.help}")
            kind = type(m).__name__.lower()
            lines.append(f"# TYPE {m.name} {kind}")
            lines.extend(m._lines())
        return "\n".join(lines) + "\n"


# --- platform-wide snapshot & dashboard -----------------------------------

def platform_snapshot(scheduler=None, autoscaler=None, loader=None) -> dict:
    """Unified view across the organs. Any component may be None."""
    snap: dict[str, Any] = {}

    if scheduler is not None:
        gpus = []
        for gid, used, total in scheduler.gpu_state():
            gpus.append({"id": gid, "used_mb": used, "total_mb": total,
                         "util_pct": (100 * used // total) if total else 0})
        snap["gpus"] = gpus
        snap["cluster_util_pct"] = (
            100 * sum(g["used_mb"] for g in gpus)
            // sum(g["total_mb"] for g in gpus)) if gpus else 0

    if autoscaler is not None:
        snap["replicas"] = autoscaler.state()
        snap["scaling"] = {
            "cold_starts": autoscaler.cold_starts,
            "scale_ups": autoscaler.scale_ups,
            "scale_downs": autoscaler.scale_downs,
            "scaled_to_zero": autoscaler.scaled_to_zero,
        }

    if loader is not None:
        # getattr defaults: works for ColdStartLoader and GpuLauncher alike
        # (the launcher has cold_loads/restores but no invalidations).
        cold_loads = getattr(loader, "cold_loads", 0)
        restores = getattr(loader, "restores", 0)
        total = restores + cold_loads
        snap["embers"] = {
            "cold_loads": cold_loads,
            "restores": restores,
            "invalidations": getattr(loader, "invalidations", 0),
            "unpark_failures": getattr(loader, "unpark_failures", 0),
            "snapshot_hit_rate": round(restores / total, 3) if total else 0.0,
        }
    return snap


def render_dashboard(snap: dict) -> str:
    lines = ["=== embers platform ==="]
    if "gpus" in snap:
        lines.append(f"cluster utilisation: {snap['cluster_util_pct']}%")
        for g in snap["gpus"]:
            lines.append(f"  {g['id']}: {g['used_mb']}/{g['total_mb']}MB "
                         f"({g['util_pct']}%)")
    if "replicas" in snap:
        rep = ", ".join(f"{m}={n}" for m, n in snap["replicas"].items()) or "—"
        lines.append(f"replicas: {rep}")
    if "scaling" in snap:
        s = snap["scaling"]
        lines.append(f"scaling: {s['cold_starts']} cold-starts, "
                     f"{s['scaled_to_zero']} scaled-to-zero")
    if "embers" in snap:
        c = snap["embers"]
        lines.append(f"cold-start: {c['restores']} restores / {c['cold_loads']} "
                     f"cold-loads  → {int(c['snapshot_hit_rate']*100)}% hit-rate "
                     f"({c['invalidations']} invalidations)")
    return "\n".join(lines)
