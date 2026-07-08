"""WormholeTraining — the qig-warp ``WormholeCache`` wired as the TRAINING basin-index.

CC2's A022 advisory + EXP-A026: ONE Fisher-Rao nearest-neighbour index over Δ⁶³ TOKEN basins serves four
training loops (not just inference READ):

  #1 nucleation gate   — a token-basin MISS (beyond ξ of every cached basin) = a genuinely distinguishable
                         NEW meaning. This is exactly the K-NUCLEATE condition ("nucleated basin has high
                         held-out d_FR to existing basins"), validated by A022 hit/miss separability. The
                         MISS SIGNAL is emitted here every step; grow_vocab ACTIVATION stays EXP-A026-gated
                         on the base demonstrably learning (§0) — wired, honestly not auto-fired.
  #2 hard-negative idx  — the persistent FR-NN index yields an anchor's FR-nearest WRONG-class basins as the
                         contrastive's hard negatives (FR-screening, not InfoNCE). Exposed via ``nearest``.
  #3 replay buffer      — the cached basins are the SLEEP/EWC-Fisher replay memory; density = consolidated
                         count. Exposed via ``sample_replay``.
  #4 data-efficiency    — a passage whose sampled token-basins are all cache HITs is geometrically redundant
                         (already seen/consolidated) → SKIP its gradient; spend the budget on MISS (novel)
                         passages. The "one dial" applied to the train loop.

KEYED ON TOKEN BASINS — deliberately NOT the passage centroid (A022 KILLED the centroid: averaging drowns
content). qig-warp's NN is brute-force O(n)/query, so the cache is CAPPED (evict-oldest) and K token-basins
are sampled per step to bound cost. None-safe: if qig-warp is absent it degrades to a no-op (novelty=1.0,
never skips), so training is never blocked.
"""
from __future__ import annotations

import os
import random
from typing import Any


class WormholeTraining:
    def __init__(self, xi: float | None = None, cap: int | None = None, k_sample: int | None = None) -> None:
        self.xi = float(os.environ.get("QIG_STUDIO_WORMHOLE_XI", xi if xi is not None else 0.35))
        self.cap = int(os.environ.get("QIG_STUDIO_WORMHOLE_CAP", cap if cap is not None else 6000))
        self.k = int(os.environ.get("QIG_STUDIO_WORMHOLE_K", k_sample if k_sample is not None else 6))
        # skip a passage only when it is OVERWHELMINGLY redundant (all sampled tokens seen) — conservative,
        # so early training (empty cache → novelty 1.0) never skips and the kernel sees everything.
        self.skip_below = float(os.environ.get("QIG_STUDIO_WORMHOLE_SKIP_BELOW", 0.05))
        self.enabled = os.environ.get("QIG_STUDIO_WORMHOLE", "1").lower() in ("1", "true", "yes")
        self.cache: Any = None
        if self.enabled:
            try:
                from qig_warp import WormholeCache
                self.cache = WormholeCache(xi=self.xi)
            except Exception:  # noqa: BLE001 — qig-warp absent → degrade to no-op (never block training)
                self.cache = None
        self.hits = 0
        self.queries = 0
        self.nucleation_signal = 0          # cumulative MISS count (#1 — novel token-meanings encountered)
        self.inserts = 0
        self._rng = random.Random(7)

    def assess(self, coords: Any) -> dict[str, Any]:
        """coords: torch ``[1, T, basin_dim]`` token basins for the passage about to be trained. Queries K
        sampled token-basins, updates the index (insert MISSes, evict to cap), and returns
        ``{novelty, skip, nucleation}``. novelty = MISS-fraction; skip (#4) fires only when novelty is below
        ``skip_below`` (near-total redundancy). Never raises — a cache fault degrades to train-everything."""
        if self.cache is None or coords is None:
            return {"novelty": 1.0, "skip": False, "nucleation": 0}
        try:
            toks = coords[0].detach().cpu().numpy()      # [T, basin_dim]
            n = int(toks.shape[0])
            if n == 0:
                return {"novelty": 1.0, "skip": False, "nucleation": 0}
            idx = self._rng.sample(range(n), min(self.k, n))
            miss = 0
            novel = []
            for i in idx:
                b = toks[i]
                self.queries += 1
                payload, _d = self.cache.query(b)
                if payload is None:
                    miss += 1
                    novel.append(b)
                else:
                    self.hits += 1
            novelty = miss / max(1, len(idx))
            self.nucleation_signal += miss               # #1 nucleation signal (grow_vocab gated elsewhere)
            for b in novel:                              # build the seen/consolidated set (#3 replay memory)
                self.cache.insert(b, {"seen": True})
                self.inserts += 1
            while len(self.cache._basins) > self.cap:            # bound the O(n) query cost (evict oldest)
                self.cache._basins.pop(0)
                self.cache._payloads.pop(0)
            return {"novelty": round(novelty, 3), "skip": novelty < self.skip_below, "nucleation": miss}
        except Exception:  # noqa: BLE001 — never let the cache break a training step
            return {"novelty": 1.0, "skip": False, "nucleation": 0}

    def nearest(self, basin: Any, k: int = 4) -> list[Any]:
        """#2 hard-negative miner: the K FR-nearest cached basins to ``basin`` (the contrastive's hard
        negatives via FR-screening). Best-effort; empty list if the cache is absent/empty."""
        if self.cache is None or not getattr(self.cache, "_basins", None):
            return []
        try:
            from qig_warp.wormhole import fisher_rao_distance
            ds = sorted(((fisher_rao_distance(basin, b), b) for b in self.cache._basins), key=lambda t: t[0])
            return [b for _d, b in ds[:k]]
        except Exception:  # noqa: BLE001
            return []

    def sample_replay(self, k: int = 8) -> list[Any]:
        """#3 SLEEP/EWC replay memory: K cached (consolidated) basins to replay during sleep."""
        if self.cache is None or not getattr(self.cache, "_basins", None):
            return []
        b = self.cache._basins
        return self._rng.sample(b, min(k, len(b)))

    def telemetry(self) -> dict[str, Any]:
        hr = self.hits / max(1, self.queries)
        return {
            "wormhole_hit_rate": round(hr, 3),
            "wormhole_density": (len(self.cache._basins) if self.cache is not None else 0),
            "wormhole_nucleation_signal": self.nucleation_signal,
            "wormhole_xi": self.xi,
        }
