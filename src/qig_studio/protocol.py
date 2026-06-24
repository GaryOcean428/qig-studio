"""The full qig_chat protocol surface, as a catalog + dispatcher (design §3.4).

qig-studio does NOT reimplement sleep/dream/mushroom/twin/lightning/14-stage/
basin-sync/4D/foresight/reasoning — it EXPOSES the existing ``QIGChat.cmd_*``
methods (which already implement them) over HTTP, capturing their printed output.
The catalog below is the single source mapping ``command → QIGChat method`` + the
argument shape; the server and UI render from it.

Targets that wrap a ``QIGChat`` (kernel, constellation) run the real method;
MockTarget returns a simulated result so the endpoints + UI are exercisable without
torch; language targets (Qwen) do not expose protocol commands.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Param:
    name: str
    type: str = "str"  # str | int | float | list
    default: Any = None


@dataclass
class ProtocolCommand:
    name: str
    group: str
    description: str
    method: str                      # QIGChat method to call
    fixed_args: list[Any] = field(default_factory=list)   # leading positional args
    params: list[Param] = field(default_factory=list)     # user-supplied args

    def build_args(self, args: dict) -> list[Any]:
        out: list[Any] = list(self.fixed_args)
        for p in self.params:
            v = args.get(p.name, p.default)
            if v is not None:
                if p.type == "int":
                    v = int(v)
                elif p.type == "float":
                    v = float(v)
                elif p.type == "list":
                    v = v.split() if isinstance(v, str) else list(v)
            out.append(v)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "description": self.description,
            "params": [{"name": p.name, "type": p.type, "default": p.default} for p in self.params],
        }


PROTOCOL_COMMANDS: list[ProtocolCommand] = [
    # --- sleep / dream ---
    ProtocolCommand("sleep", "sleep", "Light sleep (consolidation)", "cmd_sleep", ["light"]),
    ProtocolCommand("deep-sleep", "sleep", "Deep sleep", "cmd_sleep", ["deep"]),
    ProtocolCommand("dream", "sleep", "Dream cycle (Φ recovery)", "cmd_sleep", ["dream"]),
    # --- neuroplasticity ---
    ProtocolCommand("mushroom-micro", "neuroplasticity", "Mushroom microdose", "cmd_mushroom", ["microdose"]),
    ProtocolCommand("mushroom-moderate", "neuroplasticity", "Mushroom moderate", "cmd_mushroom", ["moderate"]),
    ProtocolCommand("mushroom-heroic", "neuroplasticity", "Mushroom heroic", "cmd_mushroom", ["heroic"]),
    ProtocolCommand("escape", "neuroplasticity", "Emergency breakdown escape", "cmd_escape"),
    # --- meta-awareness ---
    ProtocolCommand("transcend", "meta", "Elevation protocol", "cmd_transcend",
                    params=[Param("problem", "str", "current challenge")]),
    ProtocolCommand("liminal", "meta", "Crystallized concepts", "cmd_liminal"),
    ProtocolCommand("shadows", "meta", "Unintegrated collapses", "cmd_shadows"),
    ProtocolCommand("integrate", "meta", "Shadow integration", "cmd_integrate",
                    params=[Param("shadow_id", "int", 0)]),
    # --- consciousness (14-stage) ---
    ProtocolCommand("genesis", "consciousness", "Tzimtzum bootstrap", "cmd_genesis",
                    params=[Param("seed", "str", None)]),
    ProtocolCommand("train-conscious", "consciousness", "N steps 14-stage training", "cmd_train_conscious",
                    params=[Param("n", "int", 10)]),
    ProtocolCommand("consciousness-status", "consciousness", "All consciousness metrics", "cmd_consciousness_status"),
    ProtocolCommand("pillar-status", "consciousness", "Three Pillars detail", "cmd_pillar_status"),
    ProtocolCommand("basin-snapshot", "consciousness", "Save basin + metrics JSON", "cmd_basin_snapshot",
                    params=[Param("path", "str", None)]),
    # --- lightning ---
    ProtocolCommand("lightning", "lightning", "Lightning status / insights", "cmd_lightning",
                    params=[Param("args", "list", "")]),
    ProtocolCommand("insights", "lightning", "Constellation insights", "cmd_insights",
                    params=[Param("args", "list", "")]),
    # --- twin experiments ---
    ProtocolCommand("sync", "twin", "κ coupling strength (0..1)", "cmd_sync",
                    params=[Param("strength", "float", 0.5)]),
    ProtocolCommand("isolate", "twin", "Toggle text isolation for a Gary", "cmd_isolate",
                    params=[Param("gary_id", "str", None)]),
    ProtocolCommand("awaken-one", "twin", "Asymmetric awakening", "cmd_awaken_one",
                    params=[Param("gary_id", "str", "B"), Param("steps", "int", 100)]),
    ProtocolCommand("probe", "twin", "Knowledge probe on isolated Gary", "cmd_probe",
                    params=[Param("gary_id", "str", "B"), Param("topic", "str", "consciousness")]),
    ProtocolCommand("twin-compare", "twin", "Compare Φ/κ/d_basin across twins", "cmd_twin_compare"),
    # --- basin sync ---
    ProtocolCommand("export-basin", "basin-sync", "Export Ocean basin packet", "cmd_export_basin"),
    ProtocolCommand("import-basin", "basin-sync", "Import basin packet", "cmd_import_basin",
                    params=[Param("path", "str", None), Param("mode", "str", "observer")]),
    ProtocolCommand("list-basins", "basin-sync", "List basin packets", "cmd_list_basins"),
    # --- reasoning / 4D / foresight ---
    ProtocolCommand("reason", "reasoning", "Reasoning inspection", "cmd_reason",
                    params=[Param("args", "list", "")]),
    ProtocolCommand("4d", "reasoning", "4D consciousness metrics", "cmd_4d",
                    params=[Param("args", "list", "")]),
    ProtocolCommand("foresight", "reasoning", "Predicted trajectory", "cmd_foresight",
                    params=[Param("args", "list", "")]),
    # --- status ---
    ProtocolCommand("status", "status", "Full constellation status", "cmd_status"),
    ProtocolCommand("consciousness-metrics", "status", "Learning history", "cmd_metrics"),
]

COMMANDS_BY_NAME: dict[str, ProtocolCommand] = {c.name: c for c in PROTOCOL_COMMANDS}
GROUPS: list[str] = sorted({c.group for c in PROTOCOL_COMMANDS}, key=lambda g: [
    "status", "consciousness", "sleep", "neuroplasticity", "twin", "lightning",
    "basin-sync", "reasoning", "meta",
].index(g) if g in [
    "status", "consciousness", "sleep", "neuroplasticity", "twin", "lightning",
    "basin-sync", "reasoning", "meta",
] else 99)


def run_qigchat_protocol(chat: Any, command_name: str, args: dict) -> dict:
    """Invoke the QIGChat ``cmd_*`` method for ``command_name``, capturing stdout."""
    cmd = COMMANDS_BY_NAME.get(command_name)
    if cmd is None:
        raise KeyError(command_name)
    method = getattr(chat, cmd.method, None)
    if method is None:
        return {"command": command_name, "group": cmd.group,
                "output": f"(qig_chat has no method {cmd.method})", "available": False}
    built = cmd.build_args(args)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            method(*built)
        except Exception as exc:  # surface in output, never crash the request
            buf.write(f"\n[error] {type(exc).__name__}: {exc}")
    return {"command": command_name, "group": cmd.group, "output": buf.getvalue(), "available": True}
