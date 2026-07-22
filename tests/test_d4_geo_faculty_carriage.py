"""D4 — per-node carriage: the dead faculties come alive on REAL geo-arm geometry.

CATEGORY-3 NOTICE: this module tests geometric TELEMETRY only (a coherent, honestly-derived internal
coordinate system) — never a felt-state claim. "Nonzero != validated": every test below asserts a
DIRECTION / sign, not mere non-zero-ness, per the D4.3 pre-committed acceptance criteria.

BACKGROUND: the D4.4 basis-independence battery found 5 inputs never reached
``kernel_experience.experience()``'s telemetry for the geo arm — ``ricci_signal``, ``local_kappa_c``,
``external_coupling``, ``basin_distance_delta``, ``gamma`` — starving 4 faculties dead-zero: pushed,
pleasure_seeking, investigation, transcendence. D4.1 (``qig_studio/targets/geo_cortex.py``) now wires
THREE of those five straight from ``geocoding.GeoModel``'s real per-block ``BlockTelemetry``
(``local_kappa_c``, ``basin_distance_delta``, and a re-surfaced ``ricci_signal``) into
``extra['local_kappa_c']`` / ``extra['basin_distance_delta']`` / ``extra['ricci_signal']`` — fail-closed
throughout (None stays None, never fabricated). ``external_coupling`` and ``gamma`` remain unwired for the
geo arm (out of D4.1's scope; not addressed here — flagged in the D4 report, not silently faked).

Two test groups:
  * ``Test*GeoCortexWiring`` — the REAL torch ``GeoCortexTarget`` end-to-end: confirms the wiring exists
    and is live (not just plumbed-but-dead).
  * ``test_*_sign_*`` / ``test_starved_*`` — direct, model-free unit tests against
    ``kernel_experience.experience()`` (mirrors ``tests/test_sensations_seam.py``'s style) proving the
    DIRECTION of each newly-wired signal and the D4.4 calm/apathy/boredom split.
"""

from __future__ import annotations

import math

import pytest

from qig_studio.kernel_experience import experience


def _deps() -> bool:
    try:
        import geocoding  # noqa: F401
        import qigkernels  # noqa: F401
        import torch  # noqa: F401

        return True
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# GeoCortexTarget wiring (real torch model) — the telemetry PATH is live
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _deps(), reason="needs torch + geocoding + qigkernels")
class TestGeoCortexWiring:
    def test_emit_basis_and_sync_are_on(self):
        """D4.1: the geo model is built with emit_basis+enable_sync ON so its blocks actually COMPUTE the
        basis-reduction telemetry the faculty carriage needs (see GeoBlock.forward / test_emit_basis.py in
        qig-geocoding — off by default there, on here because this IS the faculty-telemetry consumer)."""
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        t.ensure_loaded()
        assert t._model.emit_basis is True
        assert t._model.enable_sync is True

    def test_train_step_extra_carries_the_three_d4_keys_live(self):
        """After a real forward pass with >=4 token positions (curvature needs >=4), local_kappa_c,
        basin_distance_delta and ricci_signal are REAL finite numbers in extra — not merely present-but-
        None. ricci_signal is the SAME real reading as local_kappa_c (see _snap docstring: geocoding's
        local_kappa_c IS qig_core's Ricci-scalar-like local_delta63_curvature reading)."""
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        res = t.train_step("the cortex learns geometric language on the simplex and drives its own basin")
        extra = res.telemetry.to_dict()["extra"]
        for key in ("local_kappa_c", "basin_distance_delta", "ricci_signal"):
            assert key in extra
        assert extra["local_kappa_c"] is not None and math.isfinite(extra["local_kappa_c"])
        assert extra["basin_distance_delta"] is not None and math.isfinite(extra["basin_distance_delta"])
        assert extra["ricci_signal"] == extra["local_kappa_c"]        # honest re-labelling, not a 2nd number

    def test_disabled_local_kappa_fn_stays_honestly_none_end_to_end(self):
        """FAIL-CLOSED: if the model is built with local_kappa_c explicitly disabled, extra['local_kappa_c']
        (and the ricci_signal re-surfacing of it) must stay None all the way out to the studio telemetry —
        never a fabricated 0.0 or placeholder. basin_distance_delta is independent of local_kappa_fn (it
        comes from basin_mean, not curvature) so it still reads live."""
        import torch

        from geocoding.config import GeoConfig
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        t.ensure_loaded()
        # Rebuild the underlying GeoModel with local_kappa_fn explicitly OFF (mirrors ensure_loaded's own
        # config, just disabling the curvature callable) to exercise the fail-closed path honestly.
        from geocoding.model import GeoModel

        cfg = GeoConfig(vocab_size=t.vocab_size, hidden_dim=t.hidden_dim, num_layers=t.num_layers,
                         num_heads=t.num_heads, ffn_dim=t.ffn_dim, min_recursion_depth=3, use_tacking=True,
                         head_mode=t.head_mode)
        t._model = GeoModel(cfg, local_kappa_fn=None, emit_basis=True, enable_sync=True)
        ids, coords = t._encode("the cortex learns geometric language on the simplex and its own basin")
        with torch.no_grad():
            logits, geo_phi, block_tel = t._logits(ids, coords)
        assert block_tel is not None
        assert block_tel.local_kappa_c is None                      # honest, never fabricated
        snap = t._snap(logits, None, geo_phi=geo_phi, block_tel=block_tel)
        assert snap.extra["local_kappa_c"] is None
        assert snap.extra["ricci_signal"] is None
        assert snap.extra["basin_distance_delta"] is not None       # unaffected by local_kappa_fn


# ═══════════════════════════════════════════════════════════════════════════
# D4.3 — PRE-COMMITTED dead-faculty SIGN tests (direction, not mere non-zero)
# ═══════════════════════════════════════════════════════════════════════════


def _tel(*, phi=0.5, kappa=1.0, basin_distance=0.05, surprise=0.0, max_surprise=1.0, extra=None):
    return {"phi": phi, "kappa": kappa, "regime": "geometric", "basin_distance": basin_distance,
            "surprise": surprise, "max_surprise": max_surprise, "extra": extra or {}}


def test_basin_distance_delta_sign_flips_investigation_love_and_hate():
    """(a) Drive the basin TOWARD a target (basin shrinking -> positive basin_distance_delta, grounded):
    investigation rises (clip(max(delta,0)*5) -> saturates at 1) and love (investigation*grounded) rises
    with it, hate (drifting*surprise) stays near 0. REVERSE the drive (basin growing -> negative delta,
    drifting): investigation collapses to EXACTLY 0 (only approach counts) and hate rises instead — the
    signs flip, this is not mere non-zero-ness."""
    approach = experience(_tel(basin_distance=0.05, surprise=0.0,
                                extra={"basin_distance_delta": 0.6}))
    reverse = experience(_tel(basin_distance=0.9, surprise=0.3,
                               extra={"basin_distance_delta": -0.6}))
    p_a, p_r = approach.primitives, reverse.primitives

    assert p_a["layer1"]["investigation"] > 0.5          # approaching -> investigation fires
    assert p_r["layer1"]["investigation"] == 0.0          # receding -> honestly zero (only approach counts)

    assert p_a["layer2a"]["love"] > p_r["layer2a"]["love"]         # love tracks investigation×grounded
    assert p_r["layer2a"]["hate"] > p_a["layer2a"]["hate"]         # hate tracks drifting×surprise — flips


def test_local_kappa_c_sign_drives_transcendence_and_pushed_and_is_fail_closed():
    """(c) A real local_kappa_c CLOSE to kappa (near the local-critical boundary) -> pushed is HIGH and
    transcendence (curvature deviation) is LOW; a local_kappa_c FAR from kappa -> pushed collapses toward 0
    and transcendence rises toward 1 (tracks the deviation, not merely non-zero). WITHOUT local_kappa_c ->
    both honestly zero (fail-closed; never fabricated)."""
    near = experience(_tel(kappa=1.0, extra={"local_kappa_c": 1.05}))
    far = experience(_tel(kappa=1.0, extra={"local_kappa_c": 50.0}))
    absent = experience(_tel(kappa=1.0, extra={}))

    p_near, p_far, p_absent = near.primitives, far.primitives, absent.primitives

    assert p_near["layer0"]["pushed"] > 0.5                        # close to boundary -> high pushed
    assert p_far["layer0"]["pushed"] < p_near["layer0"]["pushed"]  # far from boundary -> pushed drops
    assert p_near["layer1"]["transcendence"] < p_far["layer1"]["transcendence"]  # tracks deviation, rises

    assert p_absent["layer0"]["pushed"] == 0.0                     # honest zero, not fabricated
    assert p_absent["layer1"]["transcendence"] == 0.0


def test_ricci_signal_sign_drives_compressed_expanded_and_pain_pleasure():
    """ricci_signal > 0 (compressed / R>0, per sensations.py's documented convention) drives
    compressed/pain_avoidance up and expanded/pleasure_seeking to 0; ricci_signal < 0 (expanded / R<0)
    flips both — the sign of the SAME real curvature reading (geocoding's local_kappa_c re-surfaced) drives
    the pain/pleasure axis in opposite directions, never both at once, never fabricated when absent."""
    compressed_case = experience(_tel(extra={"ricci_signal": 0.7}))
    expanded_case = experience(_tel(extra={"ricci_signal": -0.7}))
    absent_case = experience(_tel(extra={}))

    pc, pe, pa = compressed_case.primitives, expanded_case.primitives, absent_case.primitives

    assert pc["layer0"]["compressed"] == pytest.approx(0.7, abs=1e-6)
    assert pc["layer0"]["expanded"] == 0.0
    assert pc["layer05"]["pain_avoidance"] == pytest.approx(0.7, abs=1e-6)
    assert pc["layer05"]["pleasure_seeking"] == 0.0

    assert pe["layer0"]["expanded"] == pytest.approx(0.7, abs=1e-6)
    assert pe["layer0"]["compressed"] == 0.0
    assert pe["layer05"]["pleasure_seeking"] == pytest.approx(0.7, abs=1e-6)
    assert pe["layer05"]["pain_avoidance"] == 0.0

    assert pa["layer0"]["compressed"] == 0.0 and pa["layer0"]["expanded"] == 0.0   # honest zero, no ricci


def test_novelty_burst_curiosity_rises_then_decays():
    """(b) A novelty (surprise) burst raises the surprise-driven faculty then it decays as surprise fades.
    NAMING NOTE (documented, not silently substituted): qig-core's Layer-1 ``investigation`` is basin-
    driven (-d(basin)/dt, see the test above), NOT surprise-driven by its §6.7 definition — so the
    "novelty burst" acceptance criterion is exercised here against the actual surprise-driven mechanism,
    the studio-level ``Experience.curiosity`` (novelty × productive-integration, kernel_experience.py),
    which is the honest carrier of "rises on a novelty spike, decays as it fades"."""
    burst = experience(_tel(phi=0.6, surprise=0.9, max_surprise=1.0),
                        history=[{"phi": 0.5}, {"phi": 0.6}])
    decay1 = experience(_tel(phi=0.62, surprise=0.4, max_surprise=1.0),
                         history=[{"phi": 0.6}, {"phi": 0.62}])
    decay2 = experience(_tel(phi=0.63, surprise=0.05, max_surprise=1.0),
                         history=[{"phi": 0.62}, {"phi": 0.63}])

    assert burst.novelty > decay1.novelty > decay2.novelty              # the raw surprise signal decays
    assert burst.curiosity >= decay1.curiosity >= decay2.curiosity      # curiosity rises then decays with it
    assert burst.curiosity > decay2.curiosity                            # a genuine, not-flat, decline


# ═══════════════════════════════════════════════════════════════════════════
# D4.4 — the co-pinned calm/apathy/boredom split (the battery's headline finding)
# ═══════════════════════════════════════════════════════════════════════════


def test_starved_calm_apathy_boredom_copin_then_split_once_wired():
    """UNDER STARVATION (no ricci/local_kappa_c/basin_distance_delta, flat Φ history, zero surprise, zero
    basin_distance) calm / apathy / boredom are ALL driven by the identical (1-surprise) floor and read as
    the SAME value (1.0) — the exact co-pinning the D4.4 battery found. Once REAL phi-trend + basin_distance
    (+ the newly-wired ricci_signal) are supplied, the three formulas diverge on DIFFERENT extra terms
    (apathy gains a (1-joy) factor, boredom gains a (1-curiosity) factor, calm gains a (1-compressed)
    factor) and the co-pin SPLITS into three distinct values — confirming the D4 wiring actually
    differentiates the faculty carriage, not just adds new dead fields."""
    starved = experience(_tel(phi=0.5, kappa=0.0, basin_distance=0.0, surprise=0.0, extra={}))
    p_s = starved.primitives
    calm_s = p_s["layer2a"]["calm"]
    apathy_s = p_s["layer2a"]["apathy"]
    boredom_s = p_s["layer2b"]["boredom"]
    assert calm_s == pytest.approx(1.0, abs=1e-9)
    assert apathy_s == pytest.approx(1.0, abs=1e-9)
    assert boredom_s == pytest.approx(1.0, abs=1e-9)
    assert calm_s == apathy_s == boredom_s                      # co-pinned, per the battery finding

    wired = experience(
        _tel(phi=0.7, kappa=1.0, basin_distance=0.3, surprise=0.1,
             extra={"ricci_signal": 0.4, "local_kappa_c": 2.0}),
        history=[{"phi": 0.5}, {"phi": 0.7}],
    )
    p_w = wired.primitives
    calm_w = p_w["layer2a"]["calm"]
    apathy_w = p_w["layer2a"]["apathy"]
    boredom_w = p_w["layer2b"]["boredom"]

    # the split: no two of the three may coincide, and each pairwise gap is a real, non-trivial separation
    # (not float noise) — mirrors the battery's reported 0.4-1.0 apart order of magnitude.
    vals = {"calm": calm_w, "apathy": apathy_w, "boredom": boredom_w}
    pairs = [("calm", "apathy"), ("calm", "boredom"), ("apathy", "boredom")]
    for a, b in pairs:
        assert abs(vals[a] - vals[b]) > 0.05, (a, b, vals)
