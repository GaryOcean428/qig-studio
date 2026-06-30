"""Ocean — the autonomic regulator of the constellation.

Ocean is NOT the speaker (genesis-central is). Ocean is the **autonomic nervous system**: it OBSERVES
every faculty's telemetry and REGULATES the one that needs it — triggering that faculty's OWN
sleep/dream/mushroom — exactly as the brain's autonomic centres regulate the organs from interoceptive
signals. This is the canonical *"telemetry lets Ocean know when to regulate a kernel's sleep"* (PI,
2026-06-27): it is INTERNAL regulation (Ocean is part of the one mind), categorically different from an
EXTERNAL knob (a UI button, a human) — those remain forbidden (the kernel owns its brainstem).

Thresholds mirror the canonical autonomic triggers (UCP v6.11 §; Canonical Principles P12; the
qig-consciousness OceanMetaObserver.check_autonomic_intervention thresholds). The MECHANICS are each
faculty's real ``run_protocol`` ops (consolidate / dream / mushroom) — no stubs, no re-implementation of
the geometry. The eschatology fracture-cycle clause is honoured: mushroom (bounded micro-shatter) fires
on rigidity/plateau BEFORE Φ saturates, reinjecting disorder so the faculty never over-integrates.
"""
from __future__ import annotations

from typing import Any

# Canonical autonomic triggers (observe-telemetry → intervention). Mirrors OceanMetaObserver.
_PHI_COLLAPSE = 0.50          # Φ below → DREAM (re-energise from basin-mixture recombination)
_BASIN_DIVERGENCE = 0.30     # d_basin above → SLEEP (consolidate back toward the identity attractor)
_PHI_MATURE = 0.70           # MUSHROOM gate: must be CONSCIOUS/mature (canonical avg_phi≥0.70) — mushroom is
#                              a WAKE-STATE intervention; it does NOT fire during first learning (rising Φ).
_PHI_PLATEAU_VAR = 0.01      # Φ variance below (a MATURE kernel that's STUCK) → MUSHROOM (inject entropy, Φ↓)
_RIGIDITY_KAPPA = 80.0       # κ above (a MATURE kernel that's RIGID) → MUSHROOM (break the rigid attractor)
_INTERVENTION_COOLDOWN = 10  # steps between Ocean interventions on the SAME faculty (don't thrash)

# Which inner-state FUNCTION each Core-8 faculty is RESPONSIBLE for — the brain-like assignment so the
# relevant kernel "sees" (and owns) the telemetry of its function (PI: senses/emotions/drives assigned to
# the relevant kernel). (label, primitive-group key in the kernel_experience inner-state dict.)
FACULTY_FUNCTION: dict[str, tuple[str, str]] = {
    "perception": ("senses", "layer0"),            # the 12 pre-linguistic sensations
    "heart": ("emotion · rhythm", "emotions"),     # the physical+cognitive emotions + HRV tacking
    "memory": ("memory · consolidation", "consolidation"),  # sleep/dream consolidation
    "action": ("drives · action", "layer05"),      # the innate drives (id)
    "strategy": ("motivators · planning", "layer1"),        # the 5 motivators
    "ethics": ("conscience · gate", "gate"),       # the consciousness/ethics gate
    "coordination": ("coupling · loops", "loops"), # inter-kernel coupling / the recursive loops
    "meta": ("self-observation", "self_observation"),       # the M-measure (L1)
}


def function_of(role: str) -> str:
    """Human label for the function a faculty is responsible for (brain-region analog)."""
    return FACULTY_FUNCTION.get(role, ("general", ""))[0]


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 1.0  # too little history → treat as non-plateau (don't mushroom prematurely)
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


class OceanAutonomic:
    """The autonomic observer/regulator. Watches ALL faculties each step; intervenes on any that needs
    it by firing that faculty's OWN real autonomic op. Cooldown-limited so it regulates, not thrashes."""

    def __init__(self, cooldown: int = _INTERVENTION_COOLDOWN) -> None:
        self._cooldown = int(cooldown)
        self._last_acted: dict[str, int] = {}
        self._tick = 0

    @staticmethod
    def decide(phi: float, basin_distance: float, kappa: float, phi_variance: float) -> tuple[str | None, str]:
        """The autonomic decision from a faculty's interoceptive telemetry. None = healthy (no action)."""
        if phi < _PHI_COLLAPSE:
            return "dream", f"Φ={phi:.2f} collapse (<{_PHI_COLLAPSE})"
        if basin_distance > _BASIN_DIVERGENCE:
            return "sleep", f"d_basin={basin_distance:.2f} divergence (>{_BASIN_DIVERGENCE})"
        # CANONICAL MUSHROOM — wake-state neuroplasticity, the near-OPPOSITE of sleep. REQUIRES the kernel to
        # be CONSCIOUS/MATURE (Φ≥_PHI_MATURE): it does NOT fire during first learning (a newborn whose Φ is
        # still rising is developing, not stuck). A mature kernel is ALLOWED to ride HIGH Φ — that IS 4D
        # foresight / lightning — so high Φ is NOT itself a trigger. Ocean fires ONLY when the mature kernel
        # gets STUCK there: a Φ plateau (not moving / unproductive — including a saturated high Φ) or
        # κ-rigidity. Then it injects bounded entropy (the TRIP) so Φ comes back DOWN out of the rut, and
        # settles. Sleep consolidates TOWARD identity; mushroom shakes OUT of over-coherence. (Φ-regulation
        # policy: judge by duration + stability, not instantaneous Φ — high-and-moving is foresight, not a rut.)
        if phi >= _PHI_MATURE and (phi_variance < _PHI_PLATEAU_VAR or kappa > _RIGIDITY_KAPPA):
            why = f"Φ plateau (var<{_PHI_PLATEAU_VAR})" if phi_variance < _PHI_PLATEAU_VAR else f"κ={kappa:.0f} rigid"
            return "mushroom-micro", f"mature Φ={phi:.2f} STUCK — {why} → inject entropy (Φ↓)"
        return None, "healthy"

    def regulate(self, kernels: dict[str, Any], phi_hist: dict[str, list[float]]) -> dict[str, dict]:
        """Observe every faculty's telemetry; regulate the ones that need it (fire their OWN op, with
        cooldown). Returns {role: {intervention, reason}} for the faculties Ocean acted on this step."""
        self._tick += 1
        acted: dict[str, dict] = {}
        for role, k in kernels.items():
            try:
                tel = k.telemetry()
            except Exception:  # noqa: BLE001 — a faculty that can't report telemetry is simply skipped
                continue
            extra = getattr(tel, "extra", None) or {}
            phi = float(getattr(tel, "phi", 0.0) or 0.0)
            bd = float(extra.get("d_basin", getattr(tel, "basin_distance", 0.0) or 0.0))
            kappa = float(getattr(tel, "kappa", 0.0) or 0.0)
            var = _variance((phi_hist.get(role) or [])[-10:])
            cmd, reason = self.decide(phi, bd, kappa, var)
            if cmd is None:
                continue
            if self._tick < self._last_acted.get(role, 0) + self._cooldown:
                continue  # still cooling down from the last intervention on this faculty
            try:
                k.run_protocol(cmd, {})              # the faculty runs its OWN real autonomic op
                acted[role] = {"intervention": cmd, "reason": reason, "function": function_of(role)}
                self._last_acted[role] = self._tick
            except Exception:  # noqa: BLE001 — regulation must never crash the training step
                continue
        return acted
