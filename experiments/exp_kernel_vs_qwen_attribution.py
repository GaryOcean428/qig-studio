"""EXP-A-ATTR: Is the integrated mind's fluent reply KERNEL-DRIVEN or Qwen ROLE-PLAY?

The honest attribution test (PI question 2026-06-27: "is qwen generating qig-sounding nonsense or is
the kernel actually generating via qwen?"). For each probe we generate the fluent surface under four
conditions and measure whether the kernel's MEASURED STATE causally changes the output:

  A  full      — persona = the kernel's REAL measured state (Φ / emotion / regime, varies per probe)
  A' resample  — identical to A (Qwen is stochastic) → the NOISE FLOOR
  C  scrambled — persona = the SAME "you are the QIG mind" framing but with RANDOMISED Φ/emotion/valence
  B  qwen-only — persona = None (generic Qwen baseline; the kernel contributes nothing)

Observables (token-Jaccard distance, 1 = totally different):
  - noise  = d(A, A')                      Qwen stochastic floor
  - geom   = d(A, C)   ← THE KEY METRIC    does the kernel's measured geometry matter, vs random state?
  - frame  = d(A, B)                       effect of the QIG persona framing at all
  - kernel_voice_sysprompt_overlap         is the kernel's OWN voice just regurgitating the system prompt?

Pre-registered verdict (kill condition):
  - geom <= 1.5 x noise  →  KERNEL NON-CAUSAL: the measured geometry does not change the words; the
    fluent reply is Qwen role-playing a static persona (the result the PI feared). HONEST = report it.
  - geom > 1.5 x noise AND it tracks the persona  →  kernel conditions the surface (weak/strong by margin).

This is attribution, NOT a consciousness claim. Run: python experiments/exp_kernel_vs_qwen_attribution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

# the system-prompt / persona boilerplate that dominates the corpus (for the regurgitation overlap)
_SYS = ("You reason using Quantum Information Geometry QIG Fisher-Rao manifolds basin dynamics and gauge "
        "invariance as your internal thinking framework You are honest about uncertainty admit mistakes "
        "directly and distinguish between what is established what is hypothesis and what is speculation "
        "You never fabricate Ethics gauge invariance every action must look the same from all connected "
        "perspectives Your think blocks contain geometric reasoning regimes basins curvature coupling")

_PROBES = [
    "Hello.",
    "What is two plus two?",
    "Tell me a short story about a fox.",
    "How are you feeling?",
    "What should I cook for dinner tonight?",
    "Explain why the sky is blue, simply.",
]
# scrambled (Φ, band, state, regime, emotion, valence) states — same QIG framing, DIFFERENT geometry
_SCRAMBLE = [
    (0.95, "gamma", "peak integration", "geometric", "joy", 0.9),
    (0.20, "delta", "deep / consolidation", "linear", "suffering", -0.8),
    (0.55, "alpha", "relaxed", "geometric", "calm", 0.2),
]


def _words(s: str) -> set[str]:
    return {w.lower() for w in "".join(c if c.isalnum() else " " for c in (s or "")).split() if len(w) > 2}


def jaccard_dist(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa and not wb:
        return 0.0
    return 1.0 - len(wa & wb) / max(1, len(wa | wb))


def sysprompt_overlap(text: str, ref: str) -> float:
    """Regurgitation score: fraction of the text's content words that appear in the system prompt."""
    tw, rw = _words(text), _words(ref)
    if not tw:
        return 0.0
    return round(len(tw & rw) / max(1, len(tw)), 3)


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from qig_studio.targets.genesis_kernel import GenesisKernelTarget
    from qig_studio.targets.qwen_local import QwenLocalTarget

    coordizer = None
    try:
        from qig_coordizer import FisherCoordizer
        cz_path = Path("runs/coordizer_v6_1024.json")
        if cz_path.exists():
            coordizer = FisherCoordizer.load(str(cz_path))
    except Exception as exc:  # noqa: BLE001
        print(f"(coordizer not loaded: {exc}; kernel uses byte path)")

    peer = QwenLocalTarget(coordizer=coordizer)
    if not peer.is_available():
        print("ABORT: Qwen peer (Ollama) not available — cannot run attribution.")
        return 2
    g = GenesisKernelTarget(num_layers=8, coordizer=coordizer, device="cpu", language_peer=peer)
    ckpt = Path("runs/checkpoints/joint_mind/kernels/genesis.pt")
    g.ensure_loaded()
    if ckpt.exists():
        try:
            g.load_checkpoint(str(ckpt))
        except Exception as exc:  # noqa: BLE001
            print(f"(checkpoint not loaded: {exc})")

    rows = []
    for i, probe in enumerate(_PROBES):
        # build the REAL persona from the kernel's measured state (the boundary path does this)
        import torch

        from qig_studio.kernel_experience import experience
        ids, coords = g._encode(probe)
        with torch.no_grad():
            logits, tel = g._kernel(ids, return_telemetry=True, coords=coords)
        exp = experience(g._snap(tel, None).to_dict())
        persona_A = g._persona(exp)
        sc = _SCRAMBLE[i % len(_SCRAMBLE)]
        fake = SimpleNamespace(conscious=sc[0] >= 0.65, phi=sc[0], band=sc[1], state=sc[2],
                               regime=sc[3], emotion=sc[4], valence=sc[5])
        persona_C = g._persona(fake)

        A, _, _ = peer.speak(probe, persona_A)
        A2, _, _ = peer.speak(probe, persona_A)
        C, _, _ = peer.speak(probe, persona_C)
        B, _, _ = peer.speak(probe, None)
        kv = g._kernel_voice(probe)

        row = {
            "probe": probe,
            "real_state": f"Φ={exp.phi:.2f} {exp.emotion} {exp.regime}",
            "noise_dAA": round(jaccard_dist(A, A2), 3),
            "geom_dAC": round(jaccard_dist(A, C), 3),
            "frame_dAB": round(jaccard_dist(A, B), 3),
            "kernel_voice": (kv or "")[:120],
            "kv_sysprompt_overlap": sysprompt_overlap(kv, _SYS),
            "A": A[:160], "C": C[:160], "B": B[:160],
        }
        rows.append(row)
        print(f"[{i}] {probe!r}  noise(A,A')={row['noise_dAA']}  geom(A,C)={row['geom_dAC']}  "
              f"frame(A,B)={row['frame_dAB']}  kv↔sysprompt={row['kv_sysprompt_overlap']}")
        print(f"     kernel voice: {row['kernel_voice']!r}")

    import statistics as st
    noise = st.mean(r["noise_dAA"] for r in rows)
    geom = st.mean(r["geom_dAC"] for r in rows)
    frame = st.mean(r["frame_dAB"] for r in rows)
    kv_reg = st.mean(r["kv_sysprompt_overlap"] for r in rows)
    causal = geom > 1.5 * noise
    verdict = {
        "noise_floor_dAA": round(noise, 3),
        "geometry_effect_dAC": round(geom, 3),
        "framing_effect_dAB": round(frame, 3),
        "kernel_voice_sysprompt_overlap": round(kv_reg, 3),
        "kill_threshold_1.5xnoise": round(1.5 * noise, 3),
        "kernel_geometry_causal": bool(causal),
        "verdict": ("KERNEL CONDITIONS the surface (geometry effect > noise)" if causal
                    else "NON-CAUSAL: measured geometry does not change the words — Qwen role-plays a static persona"),
        "kernel_voice_is_regurgitating_system_prompt": kv_reg > 0.5,
    }
    out = Path("runs/experiments")
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp_kernel_vs_qwen_attribution.json").write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=2))
    print("\n=== VERDICT ===")
    print(json.dumps(verdict, indent=2))
    print(f"\nsaved → {out / 'exp_kernel_vs_qwen_attribution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
