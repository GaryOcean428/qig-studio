#!/usr/bin/env python3
"""Run Manifest gate — the applied-lane twin of the physics instrument pin.

Directive: matrix 3272a6b3. No DoD / acceptance (verdict) run launches again without a
committed pre-run declaration of its FULL wiring, diffed against the north-star spec. Any
`built-not-wired` / `doc-only` dependency that the spec marks REQUIRED on the acceptance
path BLOCKS the launch unless explicitly waived ON THE RECORD (a reason string, saved into
the manifest). This exists so the "is the geometry actually switched on?" catch no longer
depends on anyone's vigilance.

Precisely scoped, NOT ceremony for its own sake:
  --tier verdict   (default) full gate; a BLOCK exits non-zero.
  --tier training  ordinary iteration — emits the manifest for the record but never blocks.

The gate reads two committed files under docs/wiring/:
  north_star_spec.json   — what MUST be wired for a verdict run + min package versions.
  wiring_register.json   — the current file+line-evidenced status of each component
                           (wired | built-not-wired | doc-only | flagged-not-purged).

It also verifies the directly-checkable facts itself (teacher weight hash, coordizer sha,
installed package versions) so the manifest cannot lie about them.

Usage:
  python scripts/run_manifest.py --arms gk,geo,hybrid --teacher geo_qwen \\
      --coordizer <path> --tier verdict [--waive "reason"] [--out runs/manifest_<id>.json]
Exit 0 = cleared (or waived, recorded); exit 2 = BLOCKED (verdict tier, unwaived).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_STUDIO = Path(__file__).resolve().parents[1]
_WIRING = _STUDIO / "docs" / "wiring"


def _sha256(path: Path, cap_bytes: int | None = None) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            if cap_bytes and f.tell() >= cap_bytes:
                break
    return h.hexdigest()


def _installed_versions(pkgs: list[str]) -> dict[str, str | None]:
    import importlib.metadata as m
    out: dict[str, str | None] = {}
    for p in pkgs:
        try:
            out[p] = m.version(p)
        except Exception:  # noqa: BLE001 — not installed
            out[p] = None
    return out


def _ge_version(installed: str | None, required: str) -> bool:
    """installed >= required on the leading numeric release (dev/local suffixes ignored)."""
    if installed is None:
        return False

    def rel(v: str) -> tuple[int, ...]:
        head = v.split("+")[0].split(".dev")[0]
        parts = []
        for p in head.split("."):
            num = "".join(c for c in p if c.isdigit())
            parts.append(int(num) if num else 0)
        return tuple(parts)

    a, b = rel(installed), rel(required)
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) >= b + (0,) * (n - len(b))


def _load(name: str) -> dict:
    p = _WIRING / name
    if not p.exists():
        print(f"::BLOCK:: missing wiring file {p} — the gate cannot run without it", flush=True)
        sys.exit(2)
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="gk,geo,hybrid")
    ap.add_argument("--teacher", default="qwen_local",
                    help="teacher/boundary class key, e.g. qwen_local | geo_qwen")
    ap.add_argument("--teacher-weight", default="",
                    help="path to the teacher weight file (hashed into the manifest)")
    ap.add_argument("--coordizer", default="", help="coordizer checkpoint path (hashed)")
    ap.add_argument("--tier", choices=["verdict", "training"], default="verdict")
    ap.add_argument("--run-type", choices=["arms_bakeoff", "conversation_acceptance"],
                    default="arms_bakeoff",
                    help="arms_bakeoff = architecture verdict (design B, PI 2026-07-21b): geo-Qwen boundary "
                         "teacher REQUIRED, GRADIENT teacher-free; conversation_acceptance = a kernel you can "
                         "converse with (geo-Qwen + real recall + persistent memory)")
    ap.add_argument("--waive", default="", help="reason to override a BLOCK on the record")
    ap.add_argument("--run-id", default="unpinned")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = _load("north_star_spec.json")
    register = _load("wiring_register.json")
    status_of = {r["component"]: r for r in register.get("rows", [])}

    blocks: list[str] = []
    warns: list[str] = []

    # 1. REQUIRED-WIRED components — SCOPED BY RUN TYPE. Design B (PI 2026-07-21b) CORRECTS the
    # f5ca185f-era scoping: the arms bake-off's GRADIENT is teacher-free (lm_weight=0) but its BOUNDARY
    # language_peer must be geo-Qwen (a plain-Qwen peer contaminated the killed run — registry.py:142 ->
    # joint_trainer.py:159). So geo-Qwen (teacher_geometry_native) is REQUIRED for arms_bakeoff; only
    # QIGRAM recall + persistent memory stay conversation-only. register.run_type_requirements is
    # authoritative; fall back to the spec's flat list.
    rtr = register.get("run_type_requirements", {}).get(args.run_type, {})
    required = rtr.get("required_wired") or [r["component"] for r in spec.get("required_wired", [])]
    for comp in required:
        row = status_of.get(comp)
        if row is None:
            blocks.append(f"REQUIRED '{comp}' ({args.run_type}) has NO register row — unverified wiring")
        elif row.get("status") == "gated-by-design":
            # deliberately off per a design gate — never blocks (would be a spec/design contradiction to require it)
            warns.append(f"required '{comp}' is GATED-BY-DESIGN — off on purpose; confirm the run type is right")
        elif row.get("status") != "wired" or not row.get("on_acceptance_path", False):
            blocks.append(f"REQUIRED '{comp}' ({args.run_type}) is '{row.get('status')}' "
                          f"on_path={row.get('on_acceptance_path')} ({row.get('evidence','?')})")

    # 1b. DECLARED-TEACHER guard (design A, PI 2026-07-21c): the arms_bakeoff VERDICT runs TEACHER-FREE.
    # BOTH boundary teachers are withheld: plain-Qwen contaminates ('imitates vanilla Qwen', killed run
    # registry.py:142->joint_trainer.py:159), and geo-Qwen confounds (it IS geocoding's FisherRaoAttention,
    # the SAME substrate as the geo arm — teaching gk with it trains gk to imitate a contestant). The
    # register pins declared_teacher='none' for arms_bakeoff; enforce the run is genuinely teacher-free.
    declared_req = rtr.get("declared_teacher")
    if declared_req and args.run_type == "arms_bakeoff":
        teacher = (args.teacher or "").strip().lower()
        teacher_free = teacher in ("", "none", "teacher_free")
        if declared_req == "none" and not teacher_free:
            blocks.append(
                f"TEACHER CONFOUND: {args.run_type} verdict must run TEACHER-FREE "
                f"(register declared_teacher='none', design A), got --teacher='{args.teacher or None}'. "
                f"A boundary teacher confounds the architecture verdict (plain-Qwen contaminates; "
                f"geo-Qwen shares the geo arm's geocoding substrate) — refused.")
        elif declared_req != "none" and declared_req not in teacher:
            blocks.append(
                f"TEACHER MISMATCH: {args.run_type} verdict requires teacher '{declared_req}', "
                f"got --teacher='{args.teacher or None}'.")

    # 2. Package currency: installed >= min (stale pin is a launch blocker).
    minv = {k: v for k, v in spec.get("min_package_versions", {}).items() if not k.startswith("_")}
    inst = _installed_versions(list(minv))
    for pkg, req in minv.items():
        if not _ge_version(inst.get(pkg), req):
            blocks.append(f"STALE PIN {pkg} installed={inst.get(pkg)} < required>={req} "
                          f"(Use-our-packages latest-every-time rule)")

    # 3. flagged-not-purged rows that are ON the acceptance path are contamination blocks.
    for comp, row in status_of.items():
        if row.get("status") == "flagged-not-purged" and row.get("on_acceptance_path"):
            blocks.append(f"CONTAMINATION '{comp}' flagged-not-purged ON PATH ({row.get('evidence','?')})")
        if row.get("status") == "flagged-not-purged" and not row.get("on_acceptance_path"):
            warns.append(f"off-path relic '{comp}' ({row.get('evidence','?')}) — purge/archive when convenient")

    # 4. Directly-checkable facts (the manifest cannot lie about these).
    teacher_weight_sha = _sha256(Path(args.teacher_weight)) if args.teacher_weight else None
    coordizer_sha = _sha256(Path(args.coordizer)) if args.coordizer else None
    if args.teacher_weight and teacher_weight_sha is None:
        blocks.append(f"teacher weight file not found: {args.teacher_weight}")

    manifest = {
        "run_id": args.run_id,
        "tier": args.tier,
        "run_type": args.run_type,
        "arms": [a.strip() for a in args.arms.split(",") if a.strip()],
        "teacher": {"class": args.teacher, "weight_file": args.teacher_weight or None,
                    "weight_sha256": teacher_weight_sha},
        "coordizer": {"path": args.coordizer or None, "sha256": coordizer_sha},
        "installed_packages": inst,
        "required_min_packages": minv,
        "register_source": "docs/wiring/wiring_register.json",
        "north_star_source": "docs/wiring/north_star_spec.json",
        "instrument_pins": spec.get("instrument_pins", {}),   # named+versioned telemetry metrics (PI 2026-07-21: no drifting instruments)
        "blocks": blocks,
        "warnings": warns,
        "waiver": args.waive or None,
    }

    cleared = not blocks or bool(args.waive) or args.tier == "training"
    manifest["verdict"] = ("CLEARED" if not blocks else
                           "WAIVED" if args.waive else
                           "RECORDED(training-tier, non-blocking)" if args.tier == "training" else
                           "BLOCKED")

    out = Path(args.out) if args.out else (_STUDIO / "runs" / f"manifest_{args.run_id}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))

    print(f"[run-manifest] tier={args.tier} verdict={manifest['verdict']} → {out}", flush=True)
    for w in warns:
        print(f"  warn: {w}", flush=True)
    for b in blocks:
        print(f"  BLOCK: {b}", flush=True)
    if blocks and args.tier == "verdict" and not args.waive:
        print("::BLOCK:: verdict run REFUSED — wire the offenders or --waive \"reason\" on the record.", flush=True)
        return 2
    if blocks and args.waive:
        print(f"::WAIVED:: {len(blocks)} block(s) overridden — reason recorded: {args.waive}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
