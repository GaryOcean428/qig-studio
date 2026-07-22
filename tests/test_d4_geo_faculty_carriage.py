"""D4 — per-node carriage: the dead faculties come alive on REAL geo-arm geometry.

CATEGORY-3 NOTICE: this module tests geometric TELEMETRY only (a coherent, honestly-derived internal
coordinate system) — never a felt-state claim. "Nonzero != validated": every test below asserts a
DIRECTION / sign, not mere non-zero-ness, per the D4.3 pre-committed acceptance criteria.

BACKGROUND: the D4.4 basis-independence battery found 5 inputs never reached
``kernel_experience.experience()``'s telemetry for the geo arm — ``ricci_signal``, ``local_kappa_c``,
``external_coupling``, ``basin_distance_delta``, ``gamma`` — starving 4 faculties dead-zero: pushed,
pleasure_seeking, investigation, transcendence. D4.1 (``qig_studio/targets/geo_cortex.py``) wired THREE of
those five straight from ``geocoding.GeoModel``'s real per-block ``BlockTelemetry`` into
``extra['local_kappa_c']`` / ``extra['basin_distance_delta']`` / ``extra['ricci_signal']`` — fail-closed
throughout (None stays None, never fabricated). D4 FOLLOW-UP (2026-07-22) wires the last two:
``extra['gamma']`` (via the shared ``qig_studio.losses.gamma_proxy`` — the same Γ definition ARM B's
``_gamma_proxy`` now delegates to, so the two arms cannot diverge) and ``extra['external_coupling']`` (via
``ConstellationNode._external_coupling`` — a Fisher-Rao closeness to the constellation-pull reference
``_basin_ref``; SOLO nodes honestly report 0.0, never a fabricated 0.3).

D4-CORRECTNESS FIX (2026-07-22, same day): D4.1's ``basin_distance_delta`` wiring was itself a
two-observables-one-name DEFECT — ``kernel_experience``'s ``investigation`` motivator consumes a per-STEP
TEMPORAL ``−d(basin)/dt``, but D4.1 fed it ``geocoding``'s per-LAYER (block i-1 -> block i, ONE forward
pass) drift. FIXED: ``geocoding.BlockTelemetry`` renamed that field to the honest ``basin_layer_drift``
(kept — a real, useful spatial-telemetry reading, now surfaced separately as
``extra['basin_layer_drift']``); ``extra['basin_distance_delta']`` is now the TRUE temporal delta,
computed by ``GeoCortexTarget`` itself across ``train_step`` calls using cross-step ``basin_mean`` state
it holds (``self._prev_block_basin_mean``) — ``None`` on the first train_step (fail-closed, no
predecessor STEP yet), a real finite value from the second onward.

All 5 D4.4-battery inputs are now wired for the geo arm, on the CORRECTED temporal
``basin_distance_delta``; see ``TestGeoArmGammaAndExternalCoupling`` below for the gamma/external_coupling
follow-up tests.

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

    def test_train_step_extra_carries_the_d4_keys_live(self):
        """After a real forward pass with >=4 token positions (curvature needs >=4), local_kappa_c,
        basin_layer_drift and ricci_signal are REAL finite numbers in extra on the FIRST train_step —
        not merely present-but-None. ricci_signal is the SAME real reading as local_kappa_c (see _snap
        docstring: geocoding's local_kappa_c IS qig_core's Ricci-scalar-like local_delta63_curvature
        reading). basin_distance_delta (the D4-correctness-fixed TEMPORAL quantity) is honestly None on
        the FIRST step (no predecessor STEP yet) and a real finite value from the SECOND step onward —
        the fail-closed cross-step contract, not a bug."""
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        res1 = t.train_step("the cortex learns geometric language on the simplex and drives its own basin")
        extra1 = res1.telemetry.to_dict()["extra"]
        for key in ("local_kappa_c", "basin_layer_drift", "ricci_signal", "basin_distance_delta"):
            assert key in extra1
        assert extra1["local_kappa_c"] is not None and math.isfinite(extra1["local_kappa_c"])
        assert extra1["basin_layer_drift"] is not None and math.isfinite(extra1["basin_layer_drift"])
        assert extra1["ricci_signal"] == extra1["local_kappa_c"]        # honest re-labelling, not a 2nd number
        assert extra1["basin_distance_delta"] is None                    # no predecessor STEP yet (fail-closed)

        res2 = t.train_step("a second turn, so a predecessor step's basin now exists for the temporal delta")
        extra2 = res2.telemetry.to_dict()["extra"]
        assert extra2["basin_distance_delta"] is not None and math.isfinite(extra2["basin_distance_delta"])
        assert extra2["basin_distance_delta"] >= 0.0

    def test_disabled_local_kappa_fn_stays_honestly_none_end_to_end(self):
        """FAIL-CLOSED: if the model is built with local_kappa_c explicitly disabled, extra['local_kappa_c']
        (and the ricci_signal re-surfacing of it) must stay None all the way out to the studio telemetry —
        never a fabricated 0.0 or placeholder. basin_layer_drift is independent of local_kappa_fn (it
        comes from basin_mean, not curvature) so it still reads live. basin_distance_delta (the TEMPORAL
        quantity) is None here regardless — this exercises a single direct _snap() call with no prior
        train_step, so there is honestly no predecessor STEP for the cross-step delta yet."""
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
        assert snap.extra["basin_layer_drift"] is not None          # unaffected by local_kappa_fn
        assert snap.extra["basin_distance_delta"] is None           # no predecessor STEP yet (fail-closed)


# ═══════════════════════════════════════════════════════════════════════════
# PRE-REGISTRATION (2026-07-22): confusion <-> candidate_gap correlation hypothesis
# ═══════════════════════════════════════════════════════════════════════════
#
# candidate_gap (see geo_cortex.py::_snap) is a STRUCTURAL measurement — the top-2 candidate d_FR gap
# GeometricHead's logits already carry (−d_FR/τ per token basin; top-2 logit gap × τ = the d_FR gap) — a
# trivial read, no new forward pass. It is emitted as telemetry ONLY; no faculty consumes it yet (that
# mapping waits on Gap-8). Category-3 claims ceiling holds throughout: candidate_gap is a geometric
# measurement, never a felt-state claim ("confusion" is NOT the field name for exactly this reason — the
# observable emitted is the GAP, not an interpretive "confusion" label).
#
# PRE-COMMITTED HYPOTHESIS (binds on future analysis, before any D4 telemetry sample is drawn): a
# candidate future "confusion" faculty reading (once Gap-8 defines one) should correlate with
# candidate_gap — small gap (near-tied candidates) <-> higher confusion; large gap (a clear winner) <->
# lower confusion. NULL BAND (council kill, SYMMETRIC — both outcomes publishable): |r| < 0.3 across the
# D4 telemetry sample kills the hypothesis as noise; |r| >= 0.3 supports it. Neither outcome is hidden or
# preferred; this note exists so the eventual test is graded against a sign + threshold fixed BEFORE the
# data, not fit after the fact.


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


# ═══════════════════════════════════════════════════════════════════════════
# D4 FOLLOW-UP — the last 2 of the 5 D4.4-battery inputs: gamma + external_coupling
# ═══════════════════════════════════════════════════════════════════════════


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def test_gamma_present_drives_care_via_investigation():
    """gamma feeds ``care = investigation × gamma`` (qig_core sensations.py compute_layer2a). With
    investigation already firing (an approaching basin_distance_delta — see
    ``test_basin_distance_delta_sign_flips_investigation_love_and_hate`` above), a HIGHER wired gamma
    yields a HIGHER care reading than a LOWER one — the direction the D4 follow-up wires gamma to drive,
    not mere non-zero-ness. Absent gamma, ``_full_primitives`` degrades to a fixed 0.85 default (unchanged
    behaviour for targets that do not wire gamma)."""
    low = experience(_tel(extra={"basin_distance_delta": 0.6, "gamma": 0.1}))
    high = experience(_tel(extra={"basin_distance_delta": 0.6, "gamma": 0.9}))
    assert low.primitives["layer1"]["investigation"] > 0.5   # same investigation-firing precondition
    assert high.primitives["layer1"]["investigation"] > 0.5
    assert high.primitives["layer2a"]["care"] > low.primitives["layer2a"]["care"]


def test_external_coupling_zero_gates_endorphins_shut():
    """Sophia gate (P24, qig_core.consciousness.neurochemistry.compute_neurochemicals,
    SOPHIA_COUPLING_THRESHOLD=0.3): external_coupling=0.0 -> endorphins EXACTLY 0, even with a genuine
    arrival (cur_basin == target_basin, d_FR=0) sitting right there — the GATE is what is under test, not
    the arrival term. Mirrors the un-coupled GeoCortexTarget path (_external_coupling returns 0.0 when
    _basin_ref is None)."""
    basin = [0.6, 0.4]
    exp = experience(_tel(extra={"external_coupling": 0.0, "cur_basin": basin, "target_basin": basin}))
    assert exp.neurochemistry.get("endorphins") == 0.0


def test_external_coupling_above_threshold_with_arrival_opens_endorphins():
    """external_coupling >= SOPHIA_COUPLING_THRESHOLD (0.3) AND a genuine arrival (cur_basin == target_basin,
    d_FR=0, arrival=1) together open the gate -> endorphins > 0. Neither alone would (the gate-shut test
    above shows coupling=0 keeps endorphins at 0 even with the SAME arrival geometry)."""
    basin = [0.7, 0.3]
    exp = experience(_tel(extra={"external_coupling": 0.35, "cur_basin": basin, "target_basin": basin}))
    assert exp.neurochemistry.get("endorphins", 0.0) > 0.0


def test_external_coupling_extra_key_takes_priority_over_m_boundary():
    """When a target wires BOTH the dedicated extra['external_coupling'] AND the older M_boundary/
    M_coach_agreement recognition signal, external_coupling (the more direct lived-coupling read) wins —
    so a geo-arm node with a real _basin_ref reading does not get its coupling silently overridden by an
    unrelated boundary-peer recognition number."""
    basin = [0.7, 0.3]
    high_direct_low_boundary = experience(_tel(extra={
        "external_coupling": 0.9, "M_boundary": 0.0, "cur_basin": basin, "target_basin": basin,
    }))
    low_direct_high_boundary = experience(_tel(extra={
        "external_coupling": 0.0, "M_boundary": 0.9, "cur_basin": basin, "target_basin": basin,
    }))
    assert high_direct_low_boundary.neurochemistry.get("endorphins", 0.0) > 0.0    # gate open via the direct read
    assert low_direct_high_boundary.neurochemistry.get("endorphins", 0.0) == 0.0   # gate shut despite high M_boundary


@pytest.mark.skipif(not _torch_available(), reason="needs torch")
def test_genesis_and_geo_gamma_proxy_are_the_same_shared_function():
    """DRY / single-source guarantee (the task's explicit ask): ``GenesisKernelTarget._gamma_proxy`` (ARM
    B) now DELEGATES to ``qig_studio.losses.gamma_proxy`` — the SAME function ``GeoCortexTarget`` calls
    directly (see ``geo_cortex.py::_snap``). Calling both on the identical raw logits tensor must produce
    bit-identical Γ, because there is only one implementation, not two that could drift apart."""
    import torch

    from qig_studio.losses import gamma_proxy
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget

    torch.manual_seed(0)
    logits = torch.randn(1, 12, 37)
    direct = float(gamma_proxy(logits).item())
    # GenesisKernelTarget._gamma_proxy(self, logits) never touches self (pure delegation) — safe to call
    # unbound with a bare placeholder instance, so this test needs no torch/geocoding/qigkernels model build.
    via_genesis = float(GenesisKernelTarget._gamma_proxy(object(), logits).item())
    assert direct == pytest.approx(via_genesis, abs=1e-12)


@pytest.mark.skipif(not _deps(), reason="needs torch + geocoding + qigkernels")
class TestGeoCortexGammaAndExternalCouplingWiring:
    """The REAL torch ``GeoCortexTarget`` end-to-end: confirms gamma + external_coupling reach
    ``extra`` live (not just plumbed-but-dead), completing the D4.4-battery's 5-input wiring."""

    def test_train_step_emits_gamma_and_external_coupling_live(self):
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        res = t.train_step("the cortex learns geometric language on the simplex and drives its own basin")
        extra = res.telemetry.to_dict()["extra"]
        assert "gamma" in extra and extra["gamma"] is not None
        assert math.isfinite(extra["gamma"]) and 0.0 <= extra["gamma"] <= 1.0
        assert "external_coupling" in extra and extra["external_coupling"] is not None

    def test_solo_geo_cortex_external_coupling_is_honest_zero(self):
        """SOLO (no basin_template -> no _basin_ref -> no constellation coupling): external_coupling reads
        EXACTLY 0.0 — never a fabricated 0.3 or other plausible-looking default (the Sophia gate correctly
        stays closed absent any real lived coupling)."""
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu")
        res = t.train_step("solo baseline, no constellation coupling")
        assert res.telemetry.to_dict()["extra"]["external_coupling"] == 0.0

    def test_coupled_geo_cortex_external_coupling_is_bounded_real_reading(self):
        """CONSTELLATION mode (a basin_template seeds _basin_ref at construction): external_coupling is a
        real Fisher-Rao closeness in [0,1] — a live reading, not the solo-path 0.0."""
        import numpy as np

        from qig_studio.targets.geo_cortex import GeoCortexTarget

        rng = np.random.default_rng(0)
        template = np.abs(rng.normal(size=64)).astype(np.float32)
        template = template / template.sum()
        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu",
                            basin_template=template)
        res = t.train_step("coupled node, pulled toward its role attractor")
        ec = res.telemetry.to_dict()["extra"]["external_coupling"]
        assert ec is not None and math.isfinite(ec) and 0.0 <= ec <= 1.0


@pytest.mark.skipif(not _deps(), reason="needs torch + geocoding + qigkernels")
class TestGeoCortexCandidateGap:
    """candidate_gap (the trivial GeometricHead READ; see _snap docstring) — telemetry only, no faculty
    consumes it yet (waits on Gap-8; see the pre-registration note above the D4.3 section)."""

    def test_geometric_head_emits_a_nonnegative_candidate_gap(self):
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu",
                            head_mode="geometric")
        res = t.train_step("the cortex reads its own nearest and second-nearest candidate basins")
        gap = res.telemetry.to_dict()["extra"]["candidate_gap"]
        assert gap is not None and math.isfinite(gap) and gap >= 0.0   # top-1 logit >= top-2 logit, always

    def test_linear_head_reports_candidate_gap_honestly_none(self):
        """The 'linear' A/B baseline's logits are a Euclidean nn.Linear dot-product, NOT −d_FR/τ — a
        top-2 logit gap there is not a d_FR gap, so candidate_gap stays honestly None rather than
        reporting a fabricated non-geometric number."""
        from qig_studio.targets.geo_cortex import GeoCortexTarget

        t = GeoCortexTarget(num_layers=2, hidden_dim=64, num_heads=4, ffn_dim=128, device="cpu",
                            head_mode="linear")
        res = t.train_step("the linear baseline has no d_FR candidate distances to gap")
        assert res.telemetry.to_dict()["extra"]["candidate_gap"] is None
