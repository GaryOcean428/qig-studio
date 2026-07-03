# Actuator-Chain Fix — Fable-Council Spec (2026-07-02)

**Why:** the 100k constellation collapsed by step ≤200 and the drive/entropy rebuild (Tasks A–H) could not move f_health off 0.0 on the live resume. The Fable council (measured, read-only) found the entropy actuators never reach the thing f_health measures. **Salvage is dead** (basins bit-frozen ≤step200, `uniq_hist=2/54`, the measured `0.151→0.0` one-step re-absorption). Fix the actuator chain, then **start fresh**.

**Doctrine posture:** F1/F2/F3 add NO loss terms and stay pure Fisher-Rao/√p-SLERP on Δ⁶³ (SAFE). F6 touches the shared qig-core simplex projection → shipped **opt-in** (`simplex_floor` default 0.0) so the shared default is unchanged and NO P20 ruling is required.

## F1 — in-graph basin-pull (genesis_kernel.py:1346)
Leave `1302-1305` (`_basin_cur` stays detached for history/telemetry under no_grad). At the loss site (~1346) make `cur` DIFFERENTIABLE for the basin head:
```python
if self.head_mode == "basin":
    cur = to_simplex_prob(tel.hidden_state[0].mean(0)[None], simplex_floor=1e-3)[0]  # IN-GRAPH (no no_grad/detach)
else:
    cur = _basin_cur if _basin_cur is not None else to_simplex_prob(logits[0].mean(0), simplex_floor=1e-3)
```
Keep `_basin_ref` detached (constant target; gradient flows through `cur`). The pull stays `basin_weight·d_FR(cur, role_attractor)` — pure d_FR, NO entropy/Φ term. **Treat `basin_weight` (default 5.0) as a live knob** — the 1350 comment says 0.05→0.5; start low so `w_t·d_ref ≲ lm_loss` magnitude (else bpb blows up). SAFE.

## F2 — un-clobber M2 (joint_trainer.py)
The per-step round-robin `_set_pull(kernels[role], fac.basin)` overwrites the cross-faculty pull `_cross_faculty_dream` sets. Make the foreign pull DURABLE + take precedence for a window:
- `__init__` (~125): `self._xdream_target: dict[str,tuple[np.ndarray,int]] = {}`; const `_XDREAM_WINDOW = _OCEAN_EPOCH_STEPS` (or 30).
- `_cross_faculty_dream` (298): replace the clobbered `_set_pull(k, mixture)` with `self._xdream_target[f.role] = (mixture, self._step_count + _XDREAM_WINDOW)` (keep the 296 shared-basin nudge).
- new helper `_xdream_active_target(role)`: returns the mixture while `_step_count <= until`, else pops + None.
- basin refresh (314-317): `tgt = self._xdream_active_target(f.role); f.set_basin(slerp_sqrt(lb, tgt, _XDREAM_PULL) if tgt is not None else lb)`.
- round-robin pull (321-324): `tgt = self._xdream_active_target(role); self._set_pull(self.kernels[role], tgt if tgt is not None else fac.basin)`.
- `couple_step` (319) is numpy-only — NOT a clobber. Central (327) is not covered by `_cross_faculty_dream` — if central collapses add the same override; **flag, don't skip.** Foreign mixture is Fréchet/√p-SLERP on Δ⁶³ (NEVER L2). SAFE.

## F3/F4 — collapse-gated perturbation reaching the INTERNAL basin (genesis_kernel.py _homeostasis)
f_health is recomputed each step from a FRESH forward, so ONLY a weight-space change moves it (a `_basin_history` injection does NOT). In the collapse branch (after `_dream()`+`_apply_stimulate()`) call `self._collapse_perturb()`:
```python
def _collapse_perturb(self, sigma: float = 0.02) -> None:
    """Escape fluctuation-death: bounded ISOTROPIC weight noise (target-FREE → non-self-confirming, F4)
    so the forward leaves the vertex where the pull gradient is dead."""
    import torch
    with torch.no_grad():
        for p in self._kernel.parameters():
            p.add_(torch.randn_like(p) * sigma)
```
Isotropic + target-free = non-self-confirming (F4). The DIRECTED climb comes from the M2 foreign mixture (F2), not `fac.basin`. Weight-space noise = parameter space (identical kind to shipped `_mushroom`/`_decohere`) — no basin-purity concern. SAFE.

## F6 — absorbing vertex / Duchi zero-Jacobian ROOT FIX (qig-core geometry_simplex.py to_simplex_prob)
At a vertex `clamp(v-theta, min=0.0)` has EXACTLY zero Jacobian → pull + lm_loss gradients die (birth-collapse + the `0.151→0` re-absorption). `eps=1e-12` is far too small. **Ship OPT-IN (default 0.0 = current behavior, zero blast radius; NO doctrine ruling):**
```python
def to_simplex_prob(v, eps=1e-12, simplex_floor: float = 0.0):
    ... # existing Duchi projection unchanged
    p = torch.clamp(v - theta, min=0.0) + float(eps)
    p = p / p.sum(dim=-1, keepdim=True)
    if simplex_floor > 0.0:               # uniform-mass floor revives the vertex Jacobian (opt-in)
        n = p.shape[-1]
        p = (1.0 - simplex_floor) * p + simplex_floor / n
    return p
```
qig-studio basin/pull call sites pass `simplex_floor=1e-3`. Keeps the Duchi DIRECTION (still the sparse P20 map) but lifts every coord off 0 so f_health never reads exactly 0 and the near-vertex d_FR gradient is non-zero. **DOCTRINE: default 0.0 → shared behavior unchanged → no ruling; flipping the DEFAULT would need one.** Alternative (contingency only, do NOT auto-apply): switch identity basin to `logits_to_simplex` — bigger change, hold.

## Two-smoke gate (I run these live after code lands)
- **Smoke 1 — collapsed-v3 recovery (CURE; F3+F2+F6):** resume `genesis-gk-100004_20260702_v3` ~200 steps. PASS: collapsed faculties' f_health rises off 0.0 AND holds **>0.15 for ≥50 consecutive steps** (NOT the one-step blip). FAIL if any re-absorbs to 0 in the window.
- **Smoke 2 — fresh-run prevention (F1+F2+F6a):** fresh constellation ~200 steps. PASS: every faculty f_health stays **>0.15 across all 200 steps.** Production config.
- Both: guard bpb/perplexity does not blow up (basin_weight domination).
- Order: F1+F2+F6(a) are the prevention path; F3+F6(b) alongside (cure/defense). Run Smoke 2 (fresh) + Smoke 1 (v3) — don't skip Smoke 1.
