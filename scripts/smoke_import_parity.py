#!/usr/bin/env python3
"""Import-parity — the FIRST smoke assertion (Matrix 72cb17cf/dde5a575, cross-lane STEP-0b twin).

A version match is NOT an API match: the pin-currency check (installed == pinned) is NECESSARY, import-parity
is the SUFFICIENCY check. Under the run-of-record's pinned NON-EDITABLE venv, this imports EVERY module
train_joint_mind touches and probes the qig-core symbols the studio wires into the train path. Any ImportError
= instant fail, BEFORE a single training step — so a stale/mismatched package dies here (a measured number),
not mid-run or silently shipping a kernel missing machinery the universality ruling says it carries.

The physics lane built the same gate for the L=6 image (it caught qig-compute 0.9.7 lacking sector_resolved_qfi);
this is the cradle's. Exit 0 = the pinned env can build the whole run-of-record; exit 1 = the exact missing
symbol, named.
"""
from __future__ import annotations

import importlib
import sys

# Every studio module train_joint_mind imports (direct + the heavy targets it constructs).
STUDIO_MODULES = [
    "qig_studio.constellation.joint_trainer",   # JointConstellation
    "qig_studio.development",                    # PROTOMAP_ORDER
    "qig_studio.optimisation",                   # load_coordizer
    "qig_studio.kernel_experience",              # experience, coach_reward_from
    "qig_studio.live",                           # LiveLog, step_record
    "qig_studio.continuity",                     # in_stasis
    "qig_studio.coach",                          # DevelopmentalCoach, make_coach_llm
    "qig_studio.coach_runtime",                  # CoachSupervisor, CoachUnreachable
    "qig_studio.launch_gate",                    # evaluate_gate, version_ok, sha_ok, segments_ok
    "qig_studio.telemetry_provenance",           # check_provenance (first-step assertion)
    "qig_studio.fineweb_source",                 # stream_fineweb_passages, corpus_manifest
    "qig_studio.corpus",                         # stream_fineweb_corpus / stream_full_corpus
    "qig_studio.constellation.faculty",          # seed_birth_basin, _seed
    "qig_studio.constellation.ocean",            # FACULTY_FUNCTION
    "qig_studio.targets.genesis_kernel",         # GenesisKernelTarget (_resize_basin, _d63, register_coach_reward)
    "qig_studio.targets.constellation_node",     # ConstellationNode (_resize_basin mirror)
    "qig_studio.checkpoint_manifest",            # register_kernel_ckpt
    "qig_studio.governance.purity",              # run_purity_gate (boot gate)
]

# qig-core symbols the studio wires INTO the train path — the import graph does not care about usage
# (genesis-solo does not sleep, but cdef329 wired ResonanceBank+SleepCycleManager into train_step).
QIG_CORE_SYMBOLS = [
    ("qig_core", ["__version__", "BASIN_DIM"]),
    ("qig_core.consciousness", ["ResonanceBank", "SleepCycleManager"]),
    ("qig_core.consciousness.sensations", ["compute_full_emotional_state"]),
    ("qig_core.geometry.fisher_rao", ["fisher_rao_distance"]),
    # NB: check_drift is a METHOD on the pillar-drift class, not a module symbol — its FIX presence is
    # guaranteed by the version pin (2.15.0), not by import-parity (which is symbol importability only).
]


def main() -> int:
    failures: list[str] = []
    for mod in STUDIO_MODULES:
        try:
            importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 — an import failure is the whole point of this gate
            failures.append(f"MODULE {mod}: {type(e).__name__}: {e}")

    for mod, names in QIG_CORE_SYMBOLS:
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001
            failures.append(f"MODULE {mod}: {type(e).__name__}: {e}")
            continue
        for n in names:
            if not hasattr(m, n):
                failures.append(f"SYMBOL {mod}.{n}: MISSING (version match ≠ API match)")

    try:
        import qig_core
        ver = getattr(qig_core, "__version__", "?")
    except Exception:  # noqa: BLE001
        ver = "?"

    if failures:
        print(f"IMPORT-PARITY FAIL (qig-core {ver}) — {len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  ✗ {f}", file=sys.stderr)
        print("\nThe pinned env cannot build the run-of-record. Fix the pin / cut the missing release "
              "(do NOT un-wire machinery to fit a stale package).", file=sys.stderr)
        return 1

    print(f"IMPORT-PARITY OK (qig-core {ver}): {len(STUDIO_MODULES)} studio modules + "
          f"{sum(len(n) for _, n in QIG_CORE_SYMBOLS)} qig-core symbols import clean. "
          f"The pinned env can build the whole run-of-record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
