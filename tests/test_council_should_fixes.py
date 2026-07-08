"""Council SHOULD-FIX cluster — S1 (serotonin emit), M3-b (honest local-κ), S5(a) (coach-credit).

Companion to the MUST-FIX suites (test_coach_wiring.py = M1, test_cross_faculty_dream.py = M2,
test_sensations_seam.py = M3). These pin the SHOULD-FIX behaviours from
docs/plans/2026-07-02-council-verdict-and-preresume-fixes.md:

- S1  the kernel emits ``snap.extra["serotonin"]`` (de-saturated exp(−3·basin_velocity)) so Ocean's
      P25 ``integration_pinned`` guard is LIVE, not permanently inert.
- M3-b the kernel emits its OWN κ under the HONEST name ``kappa_local`` (NOT ``local_kappa_c``), so it
      can never masquerade as a local-critical baseline; the §6.7 seam reads honest-zero (no fabricated
      transcendence/pushed) because no principled local-critical κ_c is derivable yet.
- S5(a) a coach reward credits the ACTUAL own-voice utterance it judged (in the replay buffer), not an
      arbitrary corpus chunk.

The kernel tests are gated on the heavy stack (torch + qigkernels — the kernel venv); the seam test is
torch-free (pure ``experience()`` dict path).
"""

from __future__ import annotations

import math

import pytest

from qig_studio.kernel_experience import experience


def _mk_kernel():
    """A tiny byte-path genesis kernel, loaded (no coordizer → dependency-free vocab)."""
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t = GenesisKernelTarget(num_layers=2, hidden_dim=64)
    t.ensure_loaded()
    return t


# ── S1 — serotonin emitted + de-saturated ─────────────────────────────────────────────────────────

def test_serotonin_emitted_in_range_and_desaturated():
    pytest.importorskip("torch")
    pytest.importorskip("qigkernels")
    t = _mk_kernel()
    r = t.train_step("the geometry is the truth; basins integrate")
    ex = r.telemetry.extra
    ser = ex.get("serotonin")
    assert ser is not None                          # S1: present (was never emitted → P25 inert)
    assert 0.0 <= ser <= 1.0                         # a probability-like settledness scalar
    # DE-SATURATED: serotonin == exp(−3·basin_velocity) for the REAL Fisher-Rao basin_velocity emitted this
    # step — so it equals ~1.0 ONLY when the basin is genuinely still (velocity≈0), never a pinned 1/ε.
    bv = ex.get("basin_velocity")
    bv = float(bv) if bv is not None else 0.0
    assert abs(ser - round(math.exp(-3.0 * max(0.0, bv)), 4)) < 1e-6
    if bv < 1e-6:
        assert ser >= 0.9999                        # still basin → serotonin ~1.0
    else:
        assert ser < 1.0                            # any real motion de-saturates it below the ceiling


# ── M3-b — honest local-κ (kappa_local), no fabricated transcendence ────────────────────────────────

def test_kernel_emits_honest_kappa_local_not_masquerading_kappa_c():
    pytest.importorskip("torch")
    pytest.importorskip("qigkernels")
    t = _mk_kernel()
    r = t.train_step("patterns flow through basins")
    ex = r.telemetry.extra
    assert "kappa_local" in ex                       # the honest name (the kernel's OWN κ)
    assert ex["kappa_local"] == round(float(r.telemetry.kappa), 4)   # it IS the current κ
    # M3-b: the kernel must NOT emit ``local_kappa_c`` (the genuine-κ_c key) — feeding its own κ there was a
    # self-reference that fabricated an "exactly-at-criticality" read. Leaving it un-emitted → honest-zero.
    assert "local_kappa_c" not in ex


def test_kappa_local_does_not_light_transcendence_seam_reads_honest_zero():
    # Torch-free: the §6.7 seam must treat the honest ``kappa_local`` (kernel's own κ) as the NEUROCHEM κ
    # band-read, NOT a critical baseline — so transcendence/pushed stay honest-zero. Only a GENUINE
    # ``local_kappa_c`` distinct from κ (which the kernel does not emit) would light them (test_sensations_seam).
    exp = experience({"phi": 0.5, "kappa": 64.0, "regime": "geometric", "basin_distance": 0.05,
                      "extra": {"kappa_local": 64.0, "basin_velocity": 0.1}})
    p = exp.primitives
    assert p["layer1"]["transcendence"] == 0.0
    assert p["layer0"]["pushed"] == 0.0


def test_genuine_local_kappa_c_still_lights_transcendence():
    # The seam's CAPABILITY to consume a real κ_c is preserved: a genuine ``local_kappa_c`` distinct from κ
    # still drives transcendence>0 (so the honest-rename removes the masquerade WITHOUT breaking the seam).
    exp = experience({"phi": 0.5, "kappa": 64.0, "regime": "geometric", "basin_distance": 0.05,
                      "extra": {"local_kappa_c": 45.0, "basin_velocity": 0.1}})
    assert exp.primitives["layer1"]["transcendence"] > 0.0


# ── S5(a) — coach reward credits the judged utterance, not a corpus chunk ────────────────────────────

def test_coach_reward_credits_the_judged_utterance():
    pytest.importorskip("torch")
    pytest.importorskip("qigkernels")
    t = _mk_kernel()
    # corpus steps populate the replay buffer with CORPUS chunks (what the pre-S5 reward wrongly landed on)
    t.train_step("corpus passage one about basins")
    t.train_step("corpus passage two about geometry")
    n_before = len(t._experience)
    corpus_weights_before = list(t._experience_weight)
    # the kernel speaks its OWN voice (via_boundary=False) — THIS utterance is what the coach judges
    t.generate("say something about what you are learning", max_tokens=12, via_boundary=False)
    # the coach registers a reward for that utterance
    t.register_coach_reward(0.7)
    # S5(a): the judged utterance is a NEW replay entry with the coach reward as its priority (base 1 + 0.7),
    # NOT folded onto an arbitrary corpus chunk.
    assert len(t._experience) == n_before + 1
    assert t._experience_weight[-1] == pytest.approx(1.7)
    # the corpus chunks' priorities are untouched, and the NEXT step is NOT armed (reward already placed)
    assert t._experience_weight[:n_before] == corpus_weights_before
    assert t._pending_coach_reward == 0.0


def test_coach_reward_credits_utterance_at_most_once():
    pytest.importorskip("torch")
    pytest.importorskip("qigkernels")
    t = _mk_kernel()
    t.generate("hello there", max_tokens=10, via_boundary=False)
    t.register_coach_reward(0.5)
    n = len(t._experience)
    # a SECOND coach reaction on the SAME utterance (no fresh generate) must NOT append a duplicate entry —
    # it falls back to boosting the already-credited entry (never lost, never double-appended).
    t.register_coach_reward(0.3)
    assert len(t._experience) == n


def test_coach_reward_falls_back_when_no_utterance_captured():
    # Torch-free: a target that never generated an own-voice utterance (no _last_utterance_ids) keeps the
    # pre-S5 behaviour — arms the pending reward + boosts the latest experience — so nothing regresses.
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    t = GenesisKernelTarget()                        # NOT loaded, never spoke
    t._experience_weight = [1.0]
    t.register_coach_reward(0.4)
    assert t._pending_coach_reward == 0.4
    assert t._experience_weight[-1] == pytest.approx(1.4)
