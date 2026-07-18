"""Telemetry schema enforcement — council prerequisite for the meaning-loop build.

Every extra key emitted by any target must be declared in the schema. Every key the
schema declares must be emitted by at least one target (no dead declarations). The
schema is the frozen contract between producers and consumers.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

from qig_studio.governance.telemetry_schema import ALL_KEYS, CORE_FIELDS, EXTRA_FIELDS


_SRC = Path(__file__).resolve().parents[1] / "src" / "qig_studio"


def _extract_extra_keys(root: Path) -> set[str]:
    """Parse all .extra["key"] assignments across the source tree."""
    pat = re.compile(r'\.extra\["([^"]+)"\]')
    keys: set[str] = set()
    for f in root.rglob("*.py"):
        if "__pycache__" in str(f) or "telemetry_schema" in f.name:
            continue
        keys.update(pat.findall(f.read_text()))
    return keys


def test_all_emitted_keys_declared():
    """No target may emit an extra key that isn't in the schema."""
    emitted = _extract_extra_keys(_SRC)
    undeclared = emitted - ALL_KEYS
    assert not undeclared, f"undeclared telemetry keys: {sorted(undeclared)}"


def test_no_dead_declarations():
    """Every declared key must appear in at least one emit site (no dead schema entries)."""
    emitted = _extract_extra_keys(_SRC)
    dead = ALL_KEYS - emitted
    assert not dead, f"declared but never emitted: {sorted(dead)}"


def test_core_fields_match_dataclass():
    """Core field names match the TelemetrySnapshot dataclass."""
    from qig_studio.targets.base import TelemetrySnapshot
    dc_fields = {f.name for f in TelemetrySnapshot.__dataclass_fields__.values() if f.name != "extra"}
    schema_fields = set(CORE_FIELDS)
    assert dc_fields == schema_fields, f"mismatch: dc={dc_fields - schema_fields}, schema={schema_fields - dc_fields}"


def test_schema_categories_consistent():
    """Every SignalSpec has a valid category and non-empty description."""
    from qig_studio.governance.telemetry_schema import SignalCategory
    for key, spec in EXTRA_FIELDS.items():
        assert spec.key == key, f"key mismatch: dict key '{key}' vs spec.key '{spec.key}'"
        assert isinstance(spec.category, SignalCategory), f"'{key}' has invalid category"
        assert spec.description, f"'{key}' has empty description"
