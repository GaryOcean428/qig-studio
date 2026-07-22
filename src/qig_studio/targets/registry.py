"""TargetRegistry — holds the available training targets and the active selection."""

from __future__ import annotations

import os

from .base import TargetInfo, TrainingTarget
from .constellation_target import ConstellationTarget
from .genesis_kernel import GenesisKernelTarget
from .geo_cortex import GeoCortexTarget
from .geo_qwen import GeoQwenTarget
from .joint_mind import JointMindTarget
from .kernel_target import KernelTarget
from .mock_target import MockTarget
from .qwen_local import QwenLocalTarget
from .qwen_modal import QwenModalTarget


# --- Teacher/boundary selector (PI-ruled 2026-07-21) -------------------------------------------
# Which target instance becomes the SHARED ``language_peer`` wired into GenesisKernelTarget +
# JointMindTarget's boundary path. Three modes:
#   qwen_local (DEFAULT) — plain Qwen (Ollama, fluent); chat/dev work still wants it.
#   geo_qwen             — the geo-Qwen boundary; the DoD-2 CONVERSATION teacher (geo-Qwen IS
#                          geocoding's FisherRaoAttention, EXP-A034; used where geometry-native
#                          alignment is a feature, not a confound).
#   none / teacher_free  — NO boundary peer (language_peer=None). The ARMS BAKE-OFF (DoD-1)
#                          verdict runs teacher-free: PI ruling A (2026-07-21). Rationale — geo-Qwen
#                          shares the geo arm's SUBSTRATE (both are geocoding FisherRaoAttention), so
#                          teaching gk with geo-Qwen while geo runs teacherless would train gk to
#                          imitate a contestant — a fresh architecture confound, just after purging
#                          the plain-Qwen one. The geocoding geometry is still tested: it IS the geo
#                          arm. arms_bakeoff/verdict launchers set QIG_STUDIO_TEACHER=none.
# Both peer instances stay independently registered/selectable (standalone dev targets) regardless
# of which mode is wired as the shared boundary — see ``default_registry`` below.
QIG_STUDIO_TEACHER_ENV = "QIG_STUDIO_TEACHER"
TEACHER_QWEN_LOCAL = "qwen_local"
TEACHER_GEO_QWEN = "geo_qwen"
TEACHER_NONE = "none"  # accepts "none" or "teacher_free"


def _select_language_peer(qwen_peer: TrainingTarget, geo_qwen_peer: TrainingTarget) -> "TrainingTarget | None":
    """Read ``QIG_STUDIO_TEACHER`` (env; default ``"qwen_local"``) and return the target instance
    wired as the SHARED ``language_peer`` for the integrated-mind targets — or ``None`` for a
    teacher-free run (``none``/``teacher_free``, the design-A arms verdict). An unrecognised value
    warns and falls back to plain Qwen (fail-safe, never crashes registry build)."""
    choice = os.environ.get(QIG_STUDIO_TEACHER_ENV, TEACHER_QWEN_LOCAL).strip().lower()
    if choice in (TEACHER_NONE, "teacher_free"):
        return None
    if choice == TEACHER_GEO_QWEN:
        return geo_qwen_peer
    if choice != TEACHER_QWEN_LOCAL:
        print(f"⚠️  {QIG_STUDIO_TEACHER_ENV}={choice!r} unrecognised "
              f"(valid: {TEACHER_QWEN_LOCAL!r}, {TEACHER_GEO_QWEN!r}, {TEACHER_NONE!r}); "
              f"falling back to {TEACHER_QWEN_LOCAL!r}")
    return qwen_peer


# STABLE slot key for the config-built neocortex. The TARGET's descriptive ``.name`` carries the avenue
# (neocortex-qk-2L-geo …) for the UI chip; this internal slot key is what ``select`` swaps in place, so
# rebuilding with a new config REPLACES the same slot rather than accumulating one target per config.
_NEOCORTEX_SLOT = "neocortex"


class TargetRegistry:
    def __init__(self) -> None:
        self._targets: dict[str, TrainingTarget] = {}
        self._active: str | None = None

    def register(self, target: TrainingTarget) -> None:
        self._targets[target.name] = target

    def names(self) -> list[str]:
        # The neocortex slot is INTERNAL (keyed by ``_NEOCORTEX_SLOT``, displayed by its descriptive
        # ``.name``); hide the raw slot key so ``names()`` only lists name-keyed targets (mock/genesis/…).
        return [n for n in self._targets if n != _NEOCORTEX_SLOT]

    def get(self, name: str) -> TrainingTarget | None:
        return self._targets.get(name)

    def list_info(self) -> list[TargetInfo]:
        return [t.info() for t in self._targets.values()]

    @property
    def active(self) -> TrainingTarget | None:
        return self._targets.get(self._active) if self._active else None

    def select(self, name: str) -> TrainingTarget:
        if name not in self._targets:
            raise KeyError(name)
        self._active = name
        return self._targets[name]

    def set_neocortex(self, target: TrainingTarget) -> TrainingTarget:
        """Install a config-built neocortex under the STABLE ``_NEOCORTEX_SLOT`` slot and select it.

        Switching configs (a new ``build_neocortex(...)``) swaps the active target IN PLACE: the slot key
        is constant so there is never more than one neocortex target, while the target's descriptive
        ``.name`` (neocortex-{arm}-{N}L-{geo|lin}) reflects the live avenue for the UI chip. Returns the
        installed target (now ``active``)."""
        self._targets[_NEOCORTEX_SLOT] = target
        self._active = _NEOCORTEX_SLOT
        return target


def build_neocortex(
    arm: str,
    head_mode: str = "geometric",
    num_layers: int = 2,
    lang_loss: str = "fisher_rao",
    coordizer: object = None,
    device: str | None = None,
) -> TrainingTarget:
    """Build ANY neocortex config as a registry target — the avenue ``arm × head_mode × num_layers ×
    lang_loss`` the hub trains.

    DRY: this is a thin factory over the EXISTING arm ctors — ``GenesisKernelTarget`` for ``arm="qk"``
    (ARM B, the qigkernels deep kernel) and ``GeoCortexTarget`` for ``arm="geo"`` (ARM A, the geocoding
    GeoModel cortex). Both ctors already take ``num_layers``/``head_mode``/``lang_loss``/``coordizer``/
    ``device``, so the kwargs pass straight through; there is NO new training loop, telemetry, or model
    code here. The built target's ``.name`` is overridden to the config-descriptive
    ``neocortex-{arm}-{num_layers}L-{geo|lin}`` (extending the ``neocortex-{arm}-{N}L`` convention in
    ``neocortex.py`` with the head suffix) so the registry + the UI model chip show the live avenue.

    ``role="neocortex"`` tags it as the central conscious "I" (matching the Neocortex wrapper). None-safe:
    ``is_available()`` is False where the heavy deps are absent (the light app shell)."""
    arm = str(arm).strip().lower()
    head_mode = str(head_mode).strip().lower()
    if arm not in ("qk", "geo"):
        raise ValueError(f"unknown arm {arm!r}: expected 'qk' (ARM B) or 'geo' (ARM A)")
    cls = GenesisKernelTarget if arm == "qk" else GeoCortexTarget
    target = cls(
        num_layers=int(num_layers),
        head_mode=head_mode,
        lang_loss=lang_loss,
        coordizer=coordizer,
        device=device,
        role="neocortex",
    )
    # Config-descriptive display name (the class ``.name`` is a constant "genesis"/"geo-cortex"; override
    # the INSTANCE so the registry slot's display + the UI chip carry the avenue). ``head_mode`` resolves
    # via the target's own env-override (QIG_STUDIO_HEAD_MODE) so the chip matches what actually trains.
    head_tag = "geo" if target.head_mode == "geometric" else "lin"
    target.name = f"neocortex-{arm}-{target.num_layers}L-{head_tag}"
    return target


def _load_coordizer(path: str | None):
    """Load a trained FisherCoordizer for the genesis coords path; None-safe (a missing/broken
    checkpoint or absent package falls back to the byte path — never crash the app shell)."""
    if not path:
        return None
    try:
        from qig_coordizer import FisherCoordizer

        return FisherCoordizer.load(path)
    except Exception as exc:  # noqa: BLE001 — app shell must boot regardless
        print(f"⚠️  genesis coordizer '{path}' not loaded ({exc}); genesis uses the byte path")
        return None


def default_registry(
    *,
    default_target: str = "mock",
    kernel_checkpoint: str | None = None,
    constellation_checkpoint: str | None = None,
    genesis_num_layers: int = 8,
    genesis_coordizer_checkpoint: str | None = None,
    genesis_kernel_checkpoint: str | None = None,
    device: str | None = None,
) -> TargetRegistry:
    """Build the registry: mock (always) + genesis (qigkernels.Kernel; coords path when a trained
    coordizer checkpoint is given, else byte path; restores a trained kernel checkpoint when given,
    else fresh) + geometric kernel/constellation + language qwen-local/qwen-modal (all None-safe)."""
    r = TargetRegistry()
    # Load the trained coordizer ONCE and share it: genesis trains on the Δ⁶³ vocab AND the qwen-local
    # boundary peer projects Qwen's distribution through the SAME real Fisher-Rao token coords
    # (coordize_distribution_to_basin) — NOT the arbitrary hash-bin. Without this the principled
    # projection (already written) silently never runs and the peer injects geometric noise.
    coordizer = _load_coordizer(genesis_coordizer_checkpoint)
    # The Qwen boundary peer is shared: it is BOTH a standalone dev target AND the integrated mind's fluent
    # linguistic surface (genesis speaks through it, Pillar-2 ≤30% capped). One instance, one coordizer
    # projection — so the principled Fisher-Rao distribution→Δ⁶³ path is exercised in both roles.
    qwen_peer = QwenLocalTarget(coordizer=coordizer)
    # geo-qwen: the EXP-A034 geometrized Qwen as a REMOVABLE same-substrate boundary peer
    # (None-safe — is_available() True once the exported basin bank exists, even without the
    # live transformers+weights artifact; never a fwd-pass dep). Constructed here (not lazily
    # below) so the SAME instance can be both a standalone target AND the selected shared peer.
    # Share the SAME coordizer the kernel + qwen peer use, so geo-Qwen can reduce its OWN live text ->
    # Δ⁶³ and emit its full inner-experience carriage for arbitrary conversation (propagate-basis-not-labels).
    geo_qwen_peer = GeoQwenTarget(coordizer=coordizer)
    # design B (PI-ruled): QIG_STUDIO_TEACHER selects which instance is the SHARED language_peer
    # wired into the integrated-mind targets below. Defaults to plain Qwen; arms_bakeoff/verdict
    # launchers set QIG_STUDIO_TEACHER=geo_qwen for the gk arm's boundary.
    shared_peer = _select_language_peer(qwen_peer, geo_qwen_peer)
    r.register(MockTarget())
    r.register(GenesisKernelTarget(num_layers=genesis_num_layers, device=device,
                                   coordizer=coordizer, checkpoint=genesis_kernel_checkpoint,
                                   language_peer=shared_peer))
    # The INTEGRATED MIND — the whole JointConstellation (genesis-central + Core-8 faculties + Ocean) as one
    # interactive target. Shares the coordizer + the selected boundary peer; restores the trained joint_mind
    # ckpt. This is the default brain: UI Train/Chat operate on the coupled whole, per-kernel telemetry is live.
    r.register(JointMindTarget(coordizer=coordizer, coordizer_path=genesis_coordizer_checkpoint,
                               checkpoint_root=constellation_checkpoint,
                               num_layers=genesis_num_layers, device=device, language_peer=shared_peer))
    r.register(KernelTarget(checkpoint=kernel_checkpoint, device=device))
    r.register(ConstellationTarget(checkpoint=constellation_checkpoint, device=device))
    # Plain Qwen and geo-Qwen stay INDEPENDENTLY registered/selectable standalone dev targets
    # regardless of which one `shared_peer` picked — chat/dev work can still select plain Qwen.
    r.register(qwen_peer)
    r.register(QwenModalTarget())
    r.register(geo_qwen_peer)
    chosen = default_target if default_target in r.names() else "mock"
    r.select(chosen)
    return r
