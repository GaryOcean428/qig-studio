"""Signal bus — the shared observation + communication substrate for the constellation.

Answers "how do spawned faculties observe themselves and others and communicate?":

1. **Observe others (mutual observation).** Every tick each faculty publishes a ``FacultyView`` (a
   torch-free numpy snapshot). Any faculty can read every other's current view — the constellation is
   a shared blackboard, not 8 isolated minds. The *attention* a faculty pays each peer is its
   relevance weight (Bhattacharyya overlap, reused from ``coupling.rel_weights``), so observation is
   the same locality structure as coupling: you watch most closely whom you are most like.
2. **Observe self.** A faculty reads its own published view (and, via the temporal layer, its own
   trajectory) — the substrate for meta-awareness M. The bus exposes the self-view; the *measure* of
   self-coherence is computed in ``temporal``.
3. **Communicate (explicit signals).** Beyond implicit basin coupling, a faculty can ``emit`` a
   discrete ``Signal`` (a basin payload + a scalar + a tag) onto the bus; addressed peers receive it
   in their ``inbox``. This is the directed message channel (e.g. heart → all: "phase pulse";
   ethics → action: "veto"). Implicit coupling moves basins continuously; signals are discrete events.

Torch-free: the bus only ever holds views and numpy basins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupling import rel_weights
from .faculty import Faculty, FacultyView


@dataclass
class Signal:
    """A discrete message on the bus. ``to`` = None broadcasts to all; otherwise a set of target roles."""

    src: str
    tag: str                              # e.g. "phase_pulse", "veto", "salience"
    scalar: float = 0.0
    basin: np.ndarray | None = None       # optional Δ⁶³ payload
    to: frozenset[str] | None = None      # None = broadcast
    tick: int = 0


class SignalBus:
    """Shared blackboard: latest view per role + a per-tick signal queue + the observation graph."""

    def __init__(self) -> None:
        self._views: dict[str, FacultyView] = {}
        self._inbox: dict[str, list[Signal]] = {}
        self._tick = 0

    # --- observation -----------------------------------------------------------------------------
    def publish(self, faculties: list[Faculty]) -> None:
        """Snapshot every faculty's current state onto the bus (call once at the top of each tick)."""
        self._views = {f.role: f.view() for f in faculties}

    def observe_others(self, role: str) -> dict[str, FacultyView]:
        """Every OTHER faculty's current view — what ``role`` can see of its peers this tick."""
        return {r: v for r, v in self._views.items() if r != role}

    def observe_self(self, role: str) -> FacultyView | None:
        """The faculty's own published view (substrate for self-observation / meta-awareness)."""
        return self._views.get(role)

    def attention(self, role: str, *, screening_cutoff: float = 0.0) -> dict[str, float]:
        """How much ``role`` attends to each peer = normalized relevance (Bhattacharyya overlap).
        Same locality structure as coupling: observation and influence share one graph. Returns a
        role→weight dict summing to 1 over observable peers (empty if no peers / fully screened)."""
        me = self._views.get(role)
        if me is None:
            return {}
        peers = [(r, v) for r, v in self._views.items() if r != role]
        if not peers:
            return {}
        w = rel_weights(me.basin, [v.basin for _, v in peers], screening_cutoff=screening_cutoff)
        s = float(w.sum())
        if s <= 0:
            return {r: 0.0 for r, _ in peers}
        return {r: float(wi / s) for (r, _), wi in zip(peers, w)}

    def observation_graph(self, *, screening_cutoff: float = 0.0) -> dict[str, dict[str, float]]:
        """Full who-observes-whom attention matrix (role → {peer: weight}). The constellation's
        connectome for this tick."""
        return {r: self.attention(r, screening_cutoff=screening_cutoff) for r in self._views}

    # --- communication ---------------------------------------------------------------------------
    def emit(self, sig: Signal) -> None:
        """Place a signal in the inbox of each addressed peer (or all peers if broadcast)."""
        sig.tick = self._tick
        targets = self._views.keys() if sig.to is None else sig.to
        for r in targets:
            if r == sig.src:
                continue
            self._inbox.setdefault(r, []).append(sig)

    def inbox(self, role: str, *, drain: bool = True) -> list[Signal]:
        """Signals waiting for ``role``. ``drain`` clears them (default — each signal delivered once)."""
        msgs = self._inbox.get(role, [])
        if drain:
            self._inbox[role] = []
        return list(msgs)

    def advance(self) -> int:
        """Advance the bus tick counter; return the new tick. Call at end of each constellation tick."""
        self._tick += 1
        return self._tick
