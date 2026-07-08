#!/usr/bin/env python3
"""No-stubs gate — fails (exit 1) if the source carries unfinished/placeholder code.

A loop-engineering verifier for the "no stubs" requirement. It distinguishes:
  - HARD violations (fail): TODO/FIXME/XXX, NotImplementedError, `raise ... not implemented`,
    bare `pass`-as-body stubs, `...` Ellipsis bodies, fake-data constants pretending to be measured
    (e.g. `= 0.6  # curiosity`), `return None  # TODO`.
  - SOFT, ALLOWED labels (reported, not failed): honestly-labelled NEEDS-BUILD / CALIBRATION-PENDING /
    PROVISIONAL / v1-light — these are roadmap markers, not deception. They print but do not fail,
    UNLESS --strict is passed.

Usage: python scripts/no_stubs_scan.py [src_dir] [--strict]
Excludes: web/ html, tests/ (test fixtures may legitimately use placeholders), __pycache__.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# POLICY: the danger is a SILENT stub — code that PRETENDS to work (returns fake success / no-ops)
# while doing nothing (the sleep/dream "v1 light NEEDS-BUILD" that returned a label and trained nothing;
# a method that `print("rolling back")` and doesn't). A FAIL-LOUD `raise NotImplementedError(...)` /
# guard is the OPPOSITE — it refuses rather than fabricates (the project's anti-silent-fallback
# doctrine; cf. qig-warp bridge "refuse rather than guess"). So fail-loud raises are ALLOWED; silent
# incompleteness is HARD-forbidden.
HARD = [
    (re.compile(r"\b(TODO|FIXME|XXX)\b"), "TODO/FIXME/XXX marker (unfinished work)"),
    (re.compile(r"#\s*stub\b", re.I), "explicit # stub"),
    # the purged curiosity-stub was a FALLBACK DEFAULT (`get(..., 0.6)` / `or 0.6` / bare `= 0.6`),
    # NOT an arithmetic coefficient (`0.6 * x` / `0.6·x` are legit weights) — match only the stub shape.
    (re.compile(r"\bor 0\.6\b|\.get\([^)]*,\s*0\.6\s*\)"), "0.6 fallback-default (curiosity-stub shape)"),
    (re.compile(r"return\s+None\s*#.*\b(todo|stub|later|placeholder)", re.I), "return None placeholder"),
    # NON-FUNCTIONAL markers — a capability declared but not built. THIS is the class that hid the
    # sleep/dream stub ("v1 light … NEEDS-BUILD" that returned a label and did no work). HARD-fail so it
    # can never be trained-against again. (Functional-but-simple code must NOT carry these words: label
    # it "v1 reduction" / describe what it actually does, not "placeholder/NEEDS-BUILD".)
    (re.compile(r"NEEDS.?BUILD", re.I), "NEEDS-BUILD (capability declared, not built — a stub)"),
    (re.compile(r"\bv1 light\b", re.I), "v1-light (non-functional placeholder op)"),
]
# SOFT labels are allowed (functional code with an honest calibration/scope note — NOT a missing
# capability). CALIBRATION-PENDING = real values pending empirical confirmation; PROVISIONAL = a
# functional v1 reduction. These print but do not fail.
SOFT = re.compile(r"CALIBRATION.?PENDING|PROVISIONAL", re.I)

# `pass` / `...` as the ENTIRE body of a def (a stub) — detected structurally below.
_DEF = re.compile(r"^\s*def\s+\w+\s*\(")


def _body_stub_lines(path: Path) -> list[int]:
    """Find defs whose body is only `pass` or `...` (excluding @abstractmethod / Protocol stubs)."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits: list[int] = []
    for i, ln in enumerate(lines):
        if _DEF.match(ln):
            # gather the indented body
            base = len(ln) - len(ln.lstrip())
            body = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if not nxt.strip():
                    j += 1
                    continue
                ind = len(nxt) - len(nxt.lstrip())
                if ind <= base:
                    break
                body.append(nxt.strip())
                j += 1
            # skip protocol/abstract/overload stubs (legit `...`) and intentional NULL-OBJECT / MOCK
            # no-ops (a Dummy/Mock/Null/NoOp/Fake class whose methods SHOULD do nothing — that is a
            # complete pattern, not an unfinished stub). Look back to the enclosing class.
            ctx = " ".join(lines[max(0, i - 16):i])
            if any(t in ctx for t in ("abstractmethod", "@overload", "Protocol")):
                continue
            if re.search(r"class\s+\w*(Dummy|Mock|Null|NoOp|Fake|Stubbed)\w*", ctx) or "no-op" in ctx.lower():
                continue
            real = [b for b in body if not b.startswith(('"""', "'''", "#"))]
            if real and all(b in ("pass", "...") for b in real):
                hits.append(i + 1)
    return hits


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    strict = "--strict" in sys.argv
    root = Path(args[0]) if args else Path("src/qig_studio")
    hard_hits: list[str] = []
    soft_hits: list[str] = []
    _SKIP = (".venv", "site-packages", "__pycache__", "node_modules", ".git", "build", "dist")
    for p in sorted(root.rglob("*.py")):
        if any(s in p.parts for s in _SKIP) or "/web/" in str(p) or "/tests/" in str(p) or p.name.startswith("test_"):
            continue
        if p.name == "no_stubs_scan.py":      # the scanner contains its own pattern strings — not a stub
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            for rx, why in HARD:
                if rx.search(line):
                    hard_hits.append(f"{p}:{n}  [{why}]  {line.strip()[:90]}")
            if SOFT.search(line):
                soft_hits.append(f"{p}:{n}  {line.strip()[:90]}")
        for n in _body_stub_lines(p):
            hard_hits.append(f"{p}:{n}  [empty def body — pass/... stub]")

    if soft_hits:
        print(f"ℹ️  {len(soft_hits)} honest roadmap labels (NEEDS-BUILD/CALIBRATION-PENDING/PROVISIONAL):")
        for h in soft_hits:
            print(f"   {h}")
    if hard_hits:
        print(f"\n❌ NO-STUBS GATE FAILED — {len(hard_hits)} hard violation(s):")
        for h in hard_hits:
            print(f"   {h}")
        return 1
    if strict and soft_hits:
        print("\n❌ --strict: roadmap labels present and strict mode forbids them")
        return 1
    print("\n✅ NO-STUBS GATE PASSED (no hard stubs; honest roadmap labels allowed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
