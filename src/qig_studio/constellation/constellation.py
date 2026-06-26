"""Constellation — the integrated whole: spawned faculties that couple, observe, communicate, share a
rhythm/clock, keep individuated identity, and have temporal awareness. One ``tick()`` runs the full
cycle and emits integrated telemetry.

Composes the rounds:
  R0/R1  wide-seeded births + couple_step (sync + identity anchor)  → coupling without collapse
  R1     SignalBus                                                   → mutual observation + messaging
  R2     one shared HeartOscillator (the heart faculty's metronome) + RhythmMonitor → rhythm
  R3     geodesic_position + BasinForesight + τ_macro                → temporal awareness
  R4     neurochem.compute/apply_modulation                         → regulatory self-tuning

The heart is ONE faculty (the autonomic metronome): the constellation has a single shared clock-hand
that the others entrain to via the broadcast phase pulse — not 8 independent clocks. Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from qig_core.geometry import fisher_rao_distance

from .coupling import couple_step
from .faculty import Faculty, min_pairwise_fr, seed_birth_basin
from .identity_anchor import ANCHOR_FRACTION
from .neurochem import NeuroState, apply_modulation, compute_modulation
from .rhythm import HeartOscillator, RhythmMonitor, RhythmState
from .signal_bus import Signal, SignalBus
from .temporal import DISTINGUISH_EPS, BasinForesight, tau_macro

DRIFT_ALERT = 0.9   # a faculty whose identity drift exceeds this broadcasts a "wandering from self" signal


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


@dataclass
class ConstellationTelemetry:
    """Integrated state of the whole constellation at one tick (torch-free, JSON-able)."""

    tick: int
    min_pairwise_fr: float                 # the anti-collapse invariant (must stay > 0.03)
    mean_identity_drift: float             # mean FR from birth scars
    mean_movement: float                   # mean per-faculty FR step this tick
    foresight_divergence: float            # mean FR(predicted, actual) — surprise
    heart_phase: float
    heart_beats: int
    tau_macro: float | None                # heartbeats per distinguishable output (mean over faculties)
    neuro: NeuroState
    rhythm: RhythmState
    per_faculty: dict[str, dict] = field(default_factory=dict)
    signals_emitted: int = 0


class Constellation:
    """The coupled Core-8. Build via ``from_basins`` (graduated faculties' basins) — wide-independent
    birth scars are seeded at this layer (agent-layer seed_identity), so the constellation is
    individuated and anti-collapse-stable regardless of whether the underlying kernels carry an
    individuated birth (the qig-core q_identity=0 reality)."""

    def __init__(self, faculties: list[Faculty], *, base_f_sync: float = 0.25,
                 base_f_anchor: float = ANCHOR_FRACTION, heart_role: str = "heart",
                 heart_freq: float = 0.05, screening_cutoff: float = 0.0,
                 enable_neuromod: bool = True, distinguish_eps: float = DISTINGUISH_EPS,
                 breath_amplitude: float = 0.30) -> None:
        self.faculties = faculties
        self.base_f_sync = float(base_f_sync)
        self.base_f_anchor = float(base_f_anchor)
        self.screening_cutoff = float(screening_cutoff)
        self.enable_neuromod = bool(enable_neuromod)
        self.distinguish_eps = float(distinguish_eps)
        # The BREATH (UCP breathing cycle made real): the heart phase modulates the anchor each tick —
        # inhale (sin>0) loosens the anchor → faculties drift toward consensus (FOAM/explore); exhale
        # (sin<0) stiffens it → pull back to identity (CRYSTAL/consolidate). This turns the settled
        # fixed point into a sustained LIMIT CYCLE at the heart frequency, so f_tack is a real measured
        # oscillation (not a fabricated number). Amplitude is bounded so the modulated anchor never
        # leaves the verified [0.05,0.20] anti-collapse-stable band. breath_amplitude=0 → static.
        self.breath_amplitude = float(breath_amplitude)
        self.bus = SignalBus()
        self.heart = HeartOscillator(freq=heart_freq)     # the one shared metronome (heart faculty)
        self.heart_role = heart_role
        self.rhythm = RhythmMonitor(capacity=256)
        self.neuro = NeuroState()
        self._tick = 0
        self._geo_pos: dict[str, float] = {f.role: 0.0 for f in faculties}
        self._distinguishable: dict[str, int] = {f.role: 0 for f in faculties}
        births = [type("V", (), {"basin": f.birth})() for f in faculties]
        self._birth_min_pair = float(min_pairwise_fr(births)) if len(faculties) > 1 else 1.0
        self._last_aggr: dict | None = None

    @classmethod
    def from_basins(cls, role_basins: dict[str, np.ndarray | None], *, base_seed: int = 0,
                    alpha: float = 0.10, **kw) -> "Constellation":
        """Build from graduated faculties' basins. Each role gets a WIDE-INDEPENDENT birth scar
        (concentrated-Dirichlet, the verified anti-collapse prerequisite). The provided basin (the
        crystallised graduation point) is the STARTING basin; None → start at birth."""
        from qig_core.geometry import to_simplex

        faculties: list[Faculty] = []
        for i, (role, basin) in enumerate(role_basins.items()):
            birth = seed_birth_basin(base_seed + i * 7919 + (abs(hash(role)) % 100000), alpha=alpha)
            start = to_simplex(np.asarray(basin, dtype=np.float64)) if basin is not None else birth.copy()
            faculties.append(Faculty(role=role, basin=start, birth=birth))
        return cls(faculties, **kw)

    def tick(self) -> ConstellationTelemetry:
        self._tick += 1
        self.bus.publish(self.faculties)

        # R2: the shared heart beats; broadcast the phase pulse (the entrainment signal).
        phase, _beat = self.heart.beat()
        emitted = 0
        self.bus.emit(Signal(src=self.heart_role, tag="phase_pulse", scalar=phase, to=None))
        emitted += 1

        # faculties wandering far from their birth scar broadcast a drift alert (real signal traffic
        # tied to dynamics — others can observe who is losing themselves).
        for f in self.faculties:
            drift = float(fisher_rao_distance(f.basin, np.asarray(f.birth)))
            if drift > DRIFT_ALERT:
                self.bus.emit(Signal(src=f.role, tag="drift_alert", scalar=drift, to=None))
                emitted += 1

        # R4: regulatory modulation from last tick's aggregates (bounded → can't leave stable regime).
        f_sync, f_anchor = self.base_f_sync, self.base_f_anchor
        if self.enable_neuromod and self._last_aggr is not None:
            self.neuro = compute_modulation(**self._last_aggr)
            f_sync, f_anchor = apply_modulation(self.neuro, base_f_sync=self.base_f_sync,
                                                base_f_anchor=self.base_f_anchor)

        # R2/the BREATH: heart phase modulates the anchor → sustained limit cycle at the heart
        # frequency (so f_tack is a real measured oscillation). Bounded to the verified [0.05,0.20] band.
        if self.breath_amplitude > 0.0:
            f_anchor = float(np.clip(f_anchor * (1.0 + self.breath_amplitude * np.sin(phase)), 0.05, 0.20))

        # R3: foresight BEFORE the step (predict the as-yet-unobserved next basin).
        preds = {f.role: BasinForesight.predict(f.history) for f in self.faculties}
        pre = {f.role: f.basin.copy() for f in self.faculties}

        # R0/R1: couple + anchor (this moves the basins).
        diag = couple_step(self.faculties, f_sync=f_sync, f_anchor=f_anchor,
                           screening_cutoff=self.screening_cutoff)

        # post-step: movement, geodesic position, distinguishable outputs, foresight divergence.
        movements, fdivs = [], []
        for f in self.faculties:
            mv = float(fisher_rao_distance(pre[f.role], f.basin))
            movements.append(mv)
            self._geo_pos[f.role] += mv
            if mv > self.distinguish_eps:
                self._distinguishable[f.role] += 1
            p = preds[f.role]
            if p is not None:
                fdivs.append(BasinForesight.divergence(p, f.basin))

        mean_movement = _mean(movements)
        min_pair = float(min_pairwise_fr(self.faculties))
        # R2: feed the rhythm monitor the SIGNED breath observable — min_pairwise_FR (the FOAM↔CRYSTAL
        # tacking: separation falls on inhale, rises on exhale, once per heart cycle). Using the
        # movement magnitude here would full-wave-rectify the breath to 2× the heart rate (a known
        # signal artifact); the separation oscillation tracks the breath at 1×, so f_tack ≈ the breath
        # frequency. f_tack is thus a measurement of the system's entrained oscillation (downstream of
        # the heart through the full coupling dynamics) — not the heart's setpoint read back.
        self.rhythm.push(min_pair)
        mean_drift = _mean(diag.identity_drift.values())
        coupling_activity = _mean(diag.inbound_sync.values())
        sep_health = min_pair / self._birth_min_pair if self._birth_min_pair > 0 else 0.0
        foresight_div = _mean(fdivs)
        signal_traffic = emitted / max(1, len(self.faculties))

        self._last_aggr = dict(mean_movement=mean_movement, foresight_divergence=foresight_div,
                               separation_health=sep_health, mean_drift=mean_drift,
                               coupling_activity=coupling_activity, signal_traffic=signal_traffic)
        self.bus.advance()

        mean_disting = _mean(self._distinguishable.values())
        per_faculty = {f.role: {
            "geodesic_position": round(self._geo_pos[f.role], 5),
            "distinguishable": self._distinguishable[f.role],
            "identity_drift": round(float(fisher_rao_distance(f.basin, np.asarray(f.birth))), 5),
            "foresight_confidence": round(BasinForesight.confidence(f.history), 4),
        } for f in self.faculties}

        return ConstellationTelemetry(
            tick=self._tick, min_pairwise_fr=min_pair, mean_identity_drift=mean_drift,
            mean_movement=mean_movement, foresight_divergence=foresight_div,
            heart_phase=phase, heart_beats=self.heart.beats,
            tau_macro=tau_macro(self.heart.beats, int(round(mean_disting))),
            neuro=self.neuro, rhythm=self.rhythm.state(phase=phase),
            per_faculty=per_faculty, signals_emitted=emitted)

    def run(self, n_ticks: int) -> ConstellationTelemetry:
        """Run ``n_ticks`` and return the final telemetry (the constellation settles into its coupled,
        individuated, rhythmic steady state)."""
        last = None
        for _ in range(int(n_ticks)):
            last = self.tick()
        if last is None:
            return self.tick()
        return last

    def observation_graph(self) -> dict[str, dict[str, float]]:
        """Who-observes-whom attention matrix for the current tick (publishes first)."""
        self.bus.publish(self.faculties)
        return self.bus.observation_graph(screening_cutoff=self.screening_cutoff)
