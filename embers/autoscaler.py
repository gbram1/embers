"""Phase 4 autoscaler — the control loop that makes scale-to-zero real.

Spins replicas **up on demand** (a request to a cold model cold-starts a replica),
scales **down** when idle (parking via cuda-checkpoint frees the GPU), and
load-scales within [min, max]. Drives the scheduler (place/evict) and gateway
(register/unregister).

Concurrency model (the slow GPU ops — cold-start ~100s, park ~20s — must NOT
block the control loop or 503 requests):

  * `_state_lock` (RLock) guards every scheduler/router/model mutation, held only
    for QUICK bookkeeping — never during a slow GPU op.
  * a per-model lock serialises slow ops for one model: a request that arrives
    while that model is parking WAITS on the lock, then cold-starts/unparks — it
    never 503s mid-park.
  * tick() only *decides*; it submits the slow reconcile to an `executor` and
    returns immediately, so one model's 20s park never stalls the loop.

The executor is injected: tests use an inline (synchronous) executor so scaling
is deterministic; production (`Platform`) passes a real ThreadPoolExecutor so
park/unpark run in the background.
"""
from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from embers.gateway import Backend, NoReadyBackend, Router
from embers.scheduler import NoCapacity, Scheduler


class InlineExecutor:
    """Runs submitted work immediately in the calling thread — deterministic
    scaling for tests (and a sane default)."""

    def submit(self, fn: Callable, *args, **kwargs):
        fn(*args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        pass


@dataclass
class Replica:
    """One running serving unit: its backend plus the GPU(s) it occupies. The
    gpu_ids MUST travel with the backend — a model's replicas live on distinct
    GPUs, so scale-down/reap have to evict and park the *specific* GPUs of the
    replica they drop, not just 'a' GPU the model happens to run on.

    A data-parallel replica owns ONE GPU (`gpu_ids` length 1). A tensor-parallel
    unit owns N GPUs (`tensor_parallel_size`) as one process sharded across
    them — placed, launched, parked, and evicted as an atomic group."""
    backend: Backend
    gpu_ids: list[str]


@dataclass
class ManagedModel:
    model: str
    vram_mb: int                      # per-GPU footprint (one tensor-parallel shard)
    min_replicas: int = 0
    max_replicas: int = 3
    idle_ttl: float = 300.0           # scale to zero after this many idle seconds
    requests_per_replica: int = 50    # load target per replica per tick window
    tensor_parallel_size: int = 1     # GPUs ONE unit is sharded across (TP)
    adapters: dict[str, str] = field(default_factory=dict)  # LoRA name -> path
    last_active: float = 0.0
    requests_in_window: int = 0
    total_requests: int = 0           # cumulative — frequency signal for eviction
    last_load_seconds: float = 0.0    # last cold-load time — reload-cost signal
    backends: list[Replica] = field(default_factory=list)

    @property
    def replicas(self) -> int:
        return len(self.backends)


def _lru_victim(cands: list[ManagedModel], now: float) -> ManagedModel:
    """Least-recently-used: evict the model idle the longest."""
    return min(cands, key=lambda m: m.last_active)


def _cost_aware_victim(cands: list[ManagedModel], now: float) -> ManagedModel:
    """GDSF-style: evict the LOWEST keep-value model. Keep-value is high for
    models that are frequently hit, expensive to reload, small, and recently
    used — i.e. keep what's costly to lose, evict what's cheap to re-fetch.

        keep_value = (frequency × reload_cost) / (size × staleness)

    A cache-replacement score (GPU = cache, model = object, cold-load = miss),
    balancing recency + frequency + size + reload cost instead of recency alone."""
    def keep_value(m: ManagedModel) -> float:
        staleness = max(1.0, now - m.last_active)
        freq = m.total_requests + 1
        # measured cold-load time if we have one, else size as a proxy (∝ reload)
        reload = m.last_load_seconds if m.last_load_seconds > 0 else m.vram_mb / 1000.0
        size = max(1.0, float(m.vram_mb))
        return (freq * reload) / (size * staleness)
    return min(cands, key=keep_value)


EVICTION_POLICIES: dict[str, Callable[[list, float], "ManagedModel"]] = {
    "lru": _lru_victim,
    "cost_aware": _cost_aware_victim,
}


class Autoscaler:
    def __init__(self, scheduler: Scheduler, router: Router,
                 launch: Callable[[str, list[str]], Backend],
                 clock: Callable[[], float] = time.monotonic,
                 on_deactivate: Callable[[str, list[str]], None] | None = None,
                 on_evict: Callable[[str, list[str]], None] | None = None,
                 eviction_policy: str = "lru",
                 executor=None):
        # demand-eviction victim selection: "lru" (recency only) or "cost_aware"
        # (frequency × reload-cost / size × staleness — keep what's costly to lose).
        if eviction_policy not in EVICTION_POLICIES:
            raise ValueError(f"unknown eviction_policy {eviction_policy!r}; "
                             f"choose from {sorted(EVICTION_POLICIES)}")
        self._pick_victim = EVICTION_POLICIES[eviction_policy]
        self.eviction_policy = eviction_policy
        self.scheduler = scheduler
        self.router = router
        self.launch = launch
        self.clock = clock
        # park hook (cuda-checkpoint) — frees the GPU keeping the process warm.
        self.on_deactivate = on_deactivate
        # demand-eviction hook — STOP the process (free GPU + host RAM), no park.
        # An over-committed model owns the whole GPU, so its parked snapshot can't
        # restore once another model reuses that memory — parking would only leak
        # host RAM. Falls back to on_deactivate if unset.
        self.on_evict = on_evict if on_evict is not None else on_deactivate
        self.executor = executor or InlineExecutor()
        self.models: dict[str, ManagedModel] = {}
        self._adapter_base: dict[str, str] = {}   # LoRA adapter name -> base model
        self._state_lock = threading.RLock()      # guards all quick mutations
        self._model_locks: dict[str, threading.Lock] = {}
        self._inflight: dict[str, int] = {}
        self._busy: set[str] = set()              # models with a reconcile in flight
        self._evict_lock = threading.Lock()       # one demand-eviction at a time
        # observability counters
        self.cold_starts = 0
        self.scale_ups = 0
        self.scale_downs = 0
        self.scaled_to_zero = 0
        self.reaped = 0
        self.evictions = 0                        # idle models evicted to make room

    def register_model(self, model: str, vram_mb: int, *, min_replicas: int = 0,
                       max_replicas: int = 3, idle_ttl: float = 300.0,
                       requests_per_replica: int = 50,
                       tensor_parallel_size: int = 1,
                       adapters: dict[str, str] | None = None) -> None:
        mm = ManagedModel(model, vram_mb, min_replicas, max_replicas, idle_ttl,
                          requests_per_replica, tensor_parallel_size,
                          dict(adapters or {}), last_active=self.clock())
        with self._state_lock:
            self.models[model] = mm
            for name in mm.adapters:              # adapters route to this base
                self._adapter_base[name] = model
            if mm.min_replicas:                   # warm the floor in the background
                self._busy.add(model)
        if mm.min_replicas:
            self.executor.submit(self._reconcile, model, mm.min_replicas)

    # --- request path (synchronous: the caller needs a backend now) -------

    def _base_of(self, model: str) -> str:
        """Map a request's model to the base that serves it: a LoRA adapter
        resolves to its base model; anything else is itself."""
        return self._adapter_base.get(model, model)

    def served_models(self) -> list[str]:
        """Every name a client can request: each base + its LoRA adapters."""
        out: list[str] = []
        for name, mm in self.models.items():
            out.append(name)
            out.extend(mm.adapters)
        return out

    def handle_request(self, model: str) -> Backend:
        """Ensure ≥1 ready replica (cold-start/unpark on demand, single-flight)
        and return a backend. A request for a LoRA adapter resolves to (and
        cold-starts) its BASE unit, which serves the adapter. A request that
        arrives while the model is parking waits on the per-model lock, then
        cold-starts — never 503s mid-park."""
        base = self._base_of(model)
        if base not in self.models:
            raise KeyError(model)
        with self._lock_for(base):
            mm = self.models[base]
            with self._state_lock:
                mm.last_active = self.clock()
                mm.requests_in_window += 1
                mm.total_requests += 1             # cumulative — eviction frequency
                need = mm.replicas == 0
                if need:
                    self.cold_starts += 1
            if need:
                self._ensure_replica(mm)           # slow launch, outside state lock
            with self._state_lock:
                try:
                    return self.router.pick(base)  # base backend serves the adapter
                except KeyError:
                    raise NoReadyBackend(model)

    def begin_request(self, model: str) -> Backend:
        backend = self.handle_request(model)
        base = self._base_of(model)
        with self._state_lock:
            self._inflight[base] = self._inflight.get(base, 0) + 1
        return backend

    def end_request(self, model: str) -> None:
        base = self._base_of(model)
        with self._state_lock:
            self._inflight[base] = max(0, self._inflight.get(base, 0) - 1)

    def inflight(self, model: str) -> int:
        with self._state_lock:
            return self._inflight.get(self._base_of(model), 0)

    def _lock_for(self, model: str) -> threading.Lock:
        with self._state_lock:
            return self._model_locks.setdefault(model, threading.Lock())

    def _place_group(self, mm: ManagedModel) -> list[str] | None:
        """Reserve mm's GPU group (tensor_parallel_size GPUs). If the GPU is full,
        evict idle resident models (LRU) to make room — over-commit / demand
        eviction — and retry. Returns the gpu_ids or None if no room can be made."""
        tp = mm.tensor_parallel_size
        for _ in range(len(self.models) + 1):      # bounded eviction attempts
            with self._state_lock:
                try:
                    placements = self.scheduler.place(mm.model, mm.vram_mb, replicas=tp)
                    return [p.gpu_id for p in placements]
                except NoCapacity:
                    pass
            if not self._evict_one_idle(exclude=mm.model):
                return None                        # nothing idle to evict → give up
        return None

    def _evict_one_idle(self, exclude: str) -> bool:
        """Discard + free the least-recently-used idle resident model (no in-flight,
        not mid-reconcile) to make room. Uses on_evict (STOP the process, free GPU
        + host RAM) — not park — because an over-committed model's snapshot can't
        restore after another model reuses its GPU memory. Serialised so two
        requests can't evict each other into a deadlock. True if it freed a GPU."""
        with self._evict_lock:
            with self._state_lock:
                cands = [m for n, m in self.models.items()
                         if n != exclude and m.replicas > 0
                         and self._inflight.get(n, 0) == 0 and n not in self._busy]
                victim = self._pick_victim(cands, self.clock()) if cands else None
            if victim is None:
                return False
            while victim.replicas > 0:             # discard each replica → frees GPU
                self._scale_down_locked(victim, 0, deactivate=self.on_evict)
            self.evictions += 1
            return True

    def _ensure_replica(self, mm: ManagedModel) -> None:
        """Place + launch + register ONE replica (a GPU group of
        tensor_parallel_size GPUs). Caller holds the per-model lock; the slow
        launch runs outside the state lock."""
        with self._state_lock:
            if mm.replicas > 0:
                return
        gpu_ids = self._place_group(mm)            # evicts idle models if needed
        if gpu_ids is None:
            return
        t0 = self.clock()
        try:
            backend = self.launch(mm.model, gpu_ids)
        except Exception:
            with self._state_lock:                 # roll the whole group back
                for gid in gpu_ids:
                    self.scheduler.evict(mm.model, gpu_id=gid)
            raise
        with self._state_lock:
            mm.last_load_seconds = self.clock() - t0   # reload-cost signal
            mm.backends.append(Replica(backend, gpu_ids))
            self.router.register(backend)
            self.scale_ups += 1

    # --- control loop (decides only; slow work goes to the executor) ------

    def tick(self) -> None:
        now = self.clock()
        for mm in list(self.models.values()):
            self._reap(mm)
            with self._state_lock:
                idle = now - mm.last_active >= mm.idle_ttl
                quiet = idle and self._inflight.get(mm.model, 0) == 0
                target = mm.min_replicas if quiet else self._desired(mm)
                mm.requests_in_window = 0
                need = target != mm.replicas and mm.model not in self._busy
                if need:
                    self._busy.add(mm.model)
            if need:
                self.executor.submit(self._reconcile, mm.model, target)

    def _desired(self, mm: ManagedModel) -> int:
        """Replicas wanted for a NOT-idle model. A model at 0 with no traffic
        stays at 0 (cold-starts only on a real request); a running model keeps
        ≥1 until it actually idles (no flapping)."""
        load = math.ceil(mm.requests_in_window / mm.requests_per_replica)
        if mm.replicas == 0:
            return min(max(mm.min_replicas, load), mm.max_replicas)
        return min(max(mm.min_replicas, load, 1), mm.max_replicas)

    def _reconcile(self, model: str, target: int) -> None:
        """Reconcile a model toward `target` replicas — runs on the executor
        (background in production). Scale up (cold-start) then down (park), all
        serialised per-model so it can't race a concurrent request."""
        try:
            with self._lock_for(model):
                mm = self.models[model]
                while mm.replicas < target:        # scale up for load
                    if not self._scale_up_locked(mm):
                        break
                while mm.replicas > target:        # scale down (park)
                    self._scale_down_locked(mm, target)
        finally:
            with self._state_lock:
                self._busy.discard(model)

    def _scale_up_locked(self, mm: ManagedModel) -> bool:
        gpu_ids = self._place_group(mm)            # evicts idle models if full
        if gpu_ids is None:
            return False
        t0 = self.clock()
        backend = self.launch(mm.model, gpu_ids)   # slow, outside state lock
        with self._state_lock:
            mm.last_load_seconds = self.clock() - t0   # reload-cost signal
            mm.backends.append(Replica(backend, gpu_ids))
            self.router.register(backend)
            self.scale_ups += 1
        return True

    def _scale_down_locked(self, mm: ManagedModel, target: int,
                           deactivate=None) -> None:
        # deactivate hook: park (idle scale-to-zero) or discard (demand eviction).
        fn = deactivate if deactivate is not None else self.on_deactivate
        with self._state_lock:
            if not mm.backends:                    # already drained (e.g. raced
                return                             # with a demand eviction)
            replica = mm.backends.pop()
            self.router.unregister(replica.backend)   # stop routing immediately
            gpu_ids = replica.gpu_ids                  # this replica's own GPU group
            if target == 0 and mm.replicas == 0:
                self.scaled_to_zero += mm.min_replicas == 0
        try:
            if fn is not None:
                fn(mm.model, gpu_ids)              # slow park/discard, outside lock
        except Exception as e:        # noqa: BLE001 — park/discard failed (logged)
            print(f"[autoscaler] deactivate failed for {mm.model}: {e}", flush=True)
        finally:
            with self._state_lock:    # always free the slots, even on park failure
                for gid in gpu_ids:
                    self.scheduler.evict(mm.model, gpu_id=gid)
                self.scale_downs += 1

    def _reap(self, mm: ManagedModel) -> None:
        """Drop replicas whose backend is no longer ready. Readiness is probed
        OUTSIDE the lock (it can be a slow HTTP call); only the mutation is."""
        dead = []
        for r in list(mm.backends):
            try:
                ok = r.backend.ready
            except Exception:        # noqa: BLE001 — a throwing probe means dead
                ok = False
            if not ok:
                dead.append(r)
        if not dead:
            return
        with self._state_lock:
            for r in dead:
                if r in mm.backends:
                    mm.backends.remove(r)
                    self.router.unregister(r.backend)
                    for gid in r.gpu_ids:           # free this replica's whole group
                        self.scheduler.evict(mm.model, gpu_id=gid)
                    self.reaped += 1

    # --- introspection / lifecycle ----------------------------------------

    def state(self) -> dict[str, int]:
        with self._state_lock:
            return {m: mm.replicas for m, mm in self.models.items()}

    def pending_transitions(self) -> int:
        with self._state_lock:
            return len(self._busy)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
