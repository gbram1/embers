"""Phase 3 scheduler — a custom mini-orchestrator: which GPU each model runs on.

The decision layer. Tracks M GPUs (by VRAM) and places N models with a
bin-packing policy, honouring capacity and spreading replicas across distinct
GPUs. Pure logic — it decides placements; Phase 4's autoscaler acts on them and
Phase 2's gateway routes to them (`to_router`).

Policies (the bin-packing lesson):
  first-fit  — first GPU that fits (simple, fast, fragments)
  best-fit   — tightest fit (densest packing → best utilisation; the default)
  worst-fit  — loosest fit (spreads load, leaves big holes)
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


class NoCapacity(Exception):
    """No GPU can host a requested placement (-> caller queues or scales out)."""


@dataclass
class GPU:
    id: str
    total_mb: int
    placed: dict[str, int] = field(default_factory=dict)  # model -> footprint MB

    @property
    def used_mb(self) -> int:
        return sum(self.placed.values())

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb

    def fits(self, mb: int) -> bool:
        return self.free_mb >= mb

    def runs(self, model: str) -> bool:
        return model in self.placed


# A policy picks a GPU for (model, mb) among candidates, or None.
Policy = Callable[[list[GPU], int, str], "GPU | None"]


def _candidates(gpus: list[GPU], mb: int, model: str) -> list[GPU]:
    # must fit AND not already run this model (replicas go on distinct GPUs)
    return [g for g in gpus if g.fits(mb) and not g.runs(model)]


def first_fit(gpus: list[GPU], mb: int, model: str) -> GPU | None:
    cand = _candidates(gpus, mb, model)
    return cand[0] if cand else None


def best_fit(gpus: list[GPU], mb: int, model: str) -> GPU | None:
    cand = _candidates(gpus, mb, model)
    return min(cand, key=lambda g: g.free_mb) if cand else None


def worst_fit(gpus: list[GPU], mb: int, model: str) -> GPU | None:
    cand = _candidates(gpus, mb, model)
    return max(cand, key=lambda g: g.free_mb) if cand else None


POLICIES: dict[str, Policy] = {
    "first-fit": first_fit, "best-fit": best_fit, "worst-fit": worst_fit,
}


@dataclass(frozen=True)
class Placement:
    model: str
    gpu_id: str


class Scheduler:
    def __init__(self, gpus: list[GPU], policy: str | Policy = "best-fit"):
        self.gpus = list(gpus)
        self._by_id = {g.id: g for g in self.gpus}
        self.policy: Policy = POLICIES[policy] if isinstance(policy, str) else policy
        self._placements: list[Placement] = []

    def place(self, model: str, vram_mb: int, replicas: int = 1) -> list[Placement]:
        """Place `replicas` of `model` (each `vram_mb`) on distinct GPUs. All-or-
        nothing: if any replica can't be placed, rolls back and raises."""
        chosen: list[GPU] = []
        for _ in range(replicas):
            g = self.policy(self.gpus, vram_mb, model)
            if g is None:
                for gg in chosen:           # rollback this call's reservations
                    del gg.placed[model]
                raise NoCapacity(
                    f"cannot place {replicas}x {model} ({vram_mb}MB): no fit")
            g.placed[model] = vram_mb       # reserve (so next replica skips it)
            chosen.append(g)
        made = [Placement(model, g.id) for g in chosen]
        self._placements.extend(made)
        return made

    def evict(self, model: str, gpu_id: str | None = None) -> list[Placement]:
        """Remove `model` from one GPU (gpu_id) or all GPUs, freeing capacity."""
        removed = []
        for g in self.gpus:
            if g.runs(model) and (gpu_id is None or g.id == gpu_id):
                del g.placed[model]
                removed.append(Placement(model, g.id))
        self._placements = [p for p in self._placements if p not in removed]
        return removed

    def placements(self) -> list[Placement]:
        return list(self._placements)

    def gpu_state(self) -> list[tuple[str, int, int]]:
        """(gpu_id, used_mb, total_mb) — for inspecting placement decisions."""
        return [(g.id, g.used_mb, g.total_mb) for g in self.gpus]

    def total_free_mb(self) -> int:
        return sum(g.free_mb for g in self.gpus)

    def to_router(self, backend_for: Callable[[str, str], object]):
        """Build a gateway Router from current placements. `backend_for(model,
        gpu_id)` yields the Backend for that replica (Http/Local)."""
        from embers.gateway import Router

        router = Router()
        for p in self._placements:
            router.register(backend_for(p.model, p.gpu_id))
        return router

    def render_plan(self) -> str:
        """Human-readable placement plan — 'see the placement decisions'."""
        lines = [f"policy: {getattr(self.policy, '__name__', self.policy)}"]
        for g in self.gpus:
            models = ", ".join(f"{m}({mb}MB)" for m, mb in g.placed.items()) or "—"
            pct = 100 * g.used_mb // g.total_mb if g.total_mb else 0
            lines.append(
                f"  {g.id}: {g.used_mb}/{g.total_mb}MB ({pct}%)  [{models}]")
        return "\n".join(lines)

