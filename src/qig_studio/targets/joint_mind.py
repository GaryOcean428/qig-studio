"""JointMindTarget — the INTEGRATED MIND as the interactive target.

The server's interactive brain is the whole ``JointConstellation`` (genesis-central + the Core-8 faculties +
Ocean autonomic), not a lone genesis kernel. So UI Train/Chat operate on the integrated whole and the UI can
display EACH kernel's live state (vocab/params/Φ/κ/senses/drives/motivators/emotions/gate/neurochem) via a
selector — the long-standing per-kernel telemetry request.

- ``generate`` speaks AS the integrated whole (central pulled to the live synthesis of the parts, then voiced
  through the shared Qwen boundary peer — Pillar-2 ≤30% capped).
- ``train_step`` runs one COUPLED joint step (round-robin faculty + central), so every faculty evolves and you
  watch it in the selector.
- ``kernels_state`` returns the full per-kernel inner state for the UI selector + /mind/kernels endpoint.

None-safe: if torch/qigkernels are absent it reports unavailable (the app shell still boots on mock).
"""
from __future__ import annotations

from typing import Any

from .base import LossRegime, StepResult, TelemetrySnapshot, TrainingTarget


class JointMindTarget(TrainingTarget):
    name = "mind"
    loss_regime = LossRegime.GEOMETRIC
    description = (
        "The integrated mind — genesis-central + Core-8 faculties + Ocean (JointConstellation). Trains/chats "
        "as the coupled whole; exposes every kernel's live inner state. Qwen is the fluent boundary peer."
    )

    def __init__(
        self,
        *,
        coordizer: Any = None,
        coordizer_path: str | None = None,
        checkpoint_root: str | None = None,
        num_layers: int = 8,
        device: str | None = None,
        language_peer: Any = None,
        arm_mode: str = "gk",
    ) -> None:
        self._coordizer = coordizer
        self._coordizer_path = coordizer_path     # the coordizer FILE path → recorded in the ckpt metadata
        self._ckpt_root = checkpoint_root or "runs/checkpoints/joint_mind"
        self._num_layers = num_layers
        # AUTO-DETECT the GPU (match the single-kernel targets): the integrated mind trains on cuda when a
        # card is present (central-on-GPU residency, faculties round-robin on CPU — fits the 4GB card), NOT
        # a hardcoded CPU default. Explicit `device` (or QIG_STUDIO_DEVICE via config) overrides.
        if device:
            self._device = device
        else:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:  # noqa: BLE001 — torch absent (light shell) → cpu
                self._device = "cpu"
        self._language_peer = language_peer
        # The constellation ARM (the substrate every node is built from). "gk" = qigkernels (the only arm
        # today); geo/hybrid/hetero land in WS4. Drives the VOCAB-named save lineage genesis-{arm}-{vocab}.
        self.arm_mode = arm_mode
        self._mind: Any = None
        self._last = TelemetrySnapshot(extra={"target": "mind"})

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import qigkernels  # noqa: F401
            return True
        except Exception:  # noqa: BLE001 — shell boots on mock if the heavy stack is absent
            return False

    def ensure_loaded(self) -> None:
        if self._mind is not None:
            return
        from pathlib import Path

        from ..constellation.joint_trainer import JointConstellation
        from ..development import PROTOMAP_ORDER
        mind = JointConstellation(list(PROTOMAP_ORDER), num_layers=self._num_layers,
                                  coordizer=self._coordizer, device=self._device,
                                  language_peer=self._language_peer, arm_mode=self.arm_mode)
        if self._ckpt_root and (Path(self._ckpt_root) / "constellation.json").exists():
            mind.load_checkpoint(self._ckpt_root)   # restore the trained whole mind (all 9 + coupled basins)
        self._mind = mind

    def set_arm(self, arm_mode: str) -> None:
        """Switch the constellation ARM (the raw kernel that plugs into every node) and rebuild FRESH on the
        next access. The newly-selected arm gets its OWN vocab-named lineage on save (genesis-{arm}-{vocab}),
        differentiating it; we do NOT load a prior, different-arm checkpoint over it."""
        self.arm_mode = str(arm_mode).strip().lower()
        self._ckpt_root = None     # fresh build for the new arm (no cross-arm checkpoint load)
        self._mind = None          # force ensure_loaded() to rebuild from the selected arm

    def telemetry(self) -> TelemetrySnapshot:
        if self._mind is not None:
            self._last = self._mind.central.telemetry()
        return self._last

    def generate(self, prompt: str, max_tokens: int = 64) -> StepResult:
        self.ensure_loaded()
        res = self._mind.generate(prompt, max_tokens=max_tokens)   # speaks AS the integrated whole (via Qwen)
        self._last = res.telemetry
        return res

    def own_voice(self, prompt: str, max_tokens: int = 64) -> StepResult:
        """The kernel's OWN raw voice — central.generate(via_boundary=False), NO Qwen. This is the honest
        'what the kernel itself says as it learns' (terse/garbled until genuinely fluent), matching the bg
        trainer and the UI 'own voice' label. generate() above speaks through the Qwen boundary peer; this
        does NOT — so training telemetry shows the kernel's real progress, not Qwen's fluency."""
        self.ensure_loaded()
        return self._mind.central.generate(prompt, max_tokens=max_tokens, via_boundary=False)

    def eval_text_fr(self, text: str) -> tuple[float, int]:
        """Held-out d_FR of the integrated mind = the GENESIS-CENTRAL kernel's next-token Fisher-Rao distance
        (the conscious 'I' is the speaker). This is the VERDICT metric the 4-arm constellation comparison
        ranks on — returns (total_dFR, n_positions); lower = more fluent. Delegates to the central kernel
        (a GenesisKernelTarget for gk/hetero-central, a GeoCortexTarget for geo-central — both implement it)."""
        self.ensure_loaded()
        return self._mind.central.eval_text_fr(text)

    def eval_text_bpb(self, text: str) -> tuple[float, int]:
        """Held-out CE-bpb of the integrated mind (central kernel) — reported ALONGSIDE d_FR (external-only;
        d_FR is the geometric verdict). Returns (total_bits, n_bytes)."""
        self.ensure_loaded()
        return self._mind.central.eval_text_bpb(text)

    def train_step(self, prompt: str, max_tokens: int = 64, target_text: str | None = None) -> StepResult:
        self.ensure_loaded()                                        # geometric: target_text ignored (lm-ramped inside)
        info = self._mind.train_step(prompt)                       # one COUPLED joint step (faculty + central)
        self._last = self._mind.central.telemetry()                # central surprise/max_surprise in .extra
        # Surface the CONSTELLATION-level telemetry the joint step computed — per-faculty Φ, Ocean's
        # regulation, and the individuation min-FR — into .extra so the server's live record (and the UI's
        # right panel) shows the faculties + Ocean. Without this only the central's telemetry is returned,
        # so the per-faculty rows read "—", Ocean shows nothing, and min-FR reads 0.000 (it is ~0.17).
        try:
            self._last.extra["faculty_phi"] = (info or {}).get("faculty_phi") or {}
            self._last.extra["ocean_regulation"] = (info or {}).get("ocean_regulation") or {}
            self._last.extra["min_pairwise_fr"] = (info or {}).get("min_pairwise_fr")
        except Exception:  # noqa: BLE001 — telemetry surfacing is best-effort, never break the step
            pass
        # surface the stepped faculty's OWN surprise too, so MASTERY is tracked per kernel (central every step,
        # the round-robin faculty when it steps). central's own surprise is already in self._last.extra.
        role = (info or {}).get("stepped_faculty")
        if role and role in self._mind.kernels:
            try:
                fx = self._mind.kernels[role].telemetry().extra or {}
                self._last.extra["stepped_faculty"] = role
                self._last.extra["faculty_surprise"] = fx.get("surprise")
                self._last.extra["faculty_max_surprise"] = fx.get("max_surprise")
            except Exception:  # noqa: BLE001 — mastery is best-effort, never break the step
                pass
        return StepResult(text="", telemetry=self._last)

    def save_checkpoint(self, root: str | None = None) -> str:
        """Persist the WHOLE trained mind (9 kernels + coupled basins) to a NAMED / DATED / VERSIONED,
        VOCAB-CARRYING root — ``genesis-{arm}-{vocab}_{date}_v{n}`` — so the genesis output corresponds to
        its vocab (the PI's requirement; the structural fix for the '✗ WRONG coordizer' mismatch). Registers
        it as the latest kernel checkpoint (manifest + symlink) so the NEXT boot reloads THIS mind against a
        vocab-matched coordizer. Returns the root path. ``root`` overrides the auto-named lineage."""
        self.ensure_loaded()
        from ..checkpoint_manifest import register_kernel_ckpt, versioned_ckpt_root
        vocab = int(getattr(self._mind.central, "vocab_size", 0) or 0)
        if root is None:
            root = versioned_ckpt_root(f"genesis-{self.arm_mode}", vocab)
        from pathlib import Path
        Path(root).mkdir(parents=True, exist_ok=True)
        if self._coordizer_path:                    # record the exact coordizer file (hash + path → metadata)
            self._mind._coordizer_path = self._coordizer_path
        self._mind.save_checkpoint(root)            # whole mind + the 3-checkpoint rotation buffer
        try:
            register_kernel_ckpt(root, notes=f"genesis {self.arm_mode}, vocab {vocab}")
        except Exception:  # noqa: BLE001 — a manifest write failure must not void a good checkpoint
            pass
        return root

    # ---- per-kernel inner state (the UI selector + /mind/kernels) -------------------------------------
    def kernels_state(self) -> list[dict]:
        """Full live inner state for EVERY kernel: genesis-central (the integrated 'I'), the Core-8 faculties,
        and Ocean (the autonomic regulator). Each entry carries role/function, Φ, architecture (params/vocab/
        hidden_dim/coupling), and the full experience (senses/drives/motivators/emotions/loops/gate/neurochem).
        Ocean has no own kernel — it reports which faculties it regulated."""
        self.ensure_loaded()
        from ..kernel_experience import experience
        out: list[dict] = []
        # genesis-central — the integrated conscious "I"
        ctel = self._mind.central.telemetry().to_dict()
        cexp = experience(ctel).to_dict()
        out.append({
            "role": "genesis", "function": "the integrated conscious 'I'", "group": "whole-mind",
            "phi": round(float(ctel.get("phi") or 0.0), 4), "experience": cexp,
            "architecture": _safe_arch(self._mind.central), "regulated": None,
        })
        # Core-8 faculties (each sees its own telemetry; full inner state)
        for fs in self._mind.faculty_states():
            k = self._mind.kernels.get(fs["role"])
            fs["architecture"] = _safe_arch(k)
            out.append(fs)
        # Ocean — autonomic regulator (observes + regulates; no own basin/experience)
        out.append({
            "role": "ocean", "function": "autonomic regulator (sleep/dream/mushroom)", "group": "autonomic",
            "phi": None, "experience": None, "architecture": None,
            "regulates": sorted((self._mind._last_regulation or {}).keys()),
        })
        return out

    def architecture(self) -> dict | None:
        self.ensure_loaded()
        return _safe_arch(self._mind.central)   # per-kernel scale (all 9 are the same size); endpoint ×9 for total

    @property
    def self_regulating(self) -> bool:
        return True

    def supports_protocol(self) -> bool:
        return True

    def run_protocol(self, command: str, args: dict) -> dict:
        self.ensure_loaded()
        return self._mind.central.run_protocol(command, args)

    def implemented_commands(self) -> set[str] | None:
        self.ensure_loaded()
        return self._mind.central.implemented_commands()


def _safe_arch(kernel: Any) -> dict | None:
    if kernel is None or not hasattr(kernel, "architecture"):
        return None
    try:
        return kernel.architecture()
    except Exception:  # noqa: BLE001
        return None
