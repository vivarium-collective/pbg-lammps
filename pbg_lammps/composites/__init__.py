"""LAMMPS composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factories** — `make_lammps_document(input_script=…)` builds a
   PBG state-dict programmatically for callers that want full control over
   the LAMMPS script + wiring. Used by `demo/demo_report.py`.

2. **Declarative `*.composite.yaml`** — sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The dashboard's
   composite explorer discovers these automatically once the package is
   installed in a workspace.

Both flavors are equivalent — pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_lammps.processes import LAMMPSProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factories (legacy / programmatic API)
# ---------------------------------------------------------------------------


def make_lammps_document(
    input_file='',
    input_script='',
    working_directory='',
    interval=1.0,
):
    """Create a composite document for a LAMMPS molecular dynamics simulation.

    Provide either a path to a LAMMPS .in file (`input_file`) or an
    inline script (`input_script`). `run` / `rerun` commands in the
    script are stripped — the orchestrator drives integration based
    on `interval`.

    Returns a document dict ready for use with Composite().
    """
    if not input_file and not input_script:
        raise ValueError(
            'make_lammps_document requires input_file or input_script')

    config = {
        'input_file': input_file,
        'input_script': input_script,
        'working_directory': working_directory,
    }

    return {
        'lammps': {
            '_type': 'process',
            'address': 'local:LAMMPSProcess',
            'config': config,
            'interval': interval,
            'inputs': {},
            'outputs': {
                'temperature': ['stores', 'temperature'],
                'potential_energy': ['stores', 'potential_energy'],
                'kinetic_energy': ['stores', 'kinetic_energy'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'num_atoms': ['stores', 'num_atoms'],
                'positions': ['stores', 'positions'],
                'velocities': ['stores', 'velocities'],
                'atom_types': ['stores', 'atom_types'],
                'volume': ['stores', 'volume'],
                'pxx': ['stores', 'pxx'],
                'pyy': ['stores', 'pyy'],
                'pzz': ['stores', 'pzz'],
                'box_dimensions': ['stores', 'box_dimensions'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'temperature': 'float',
                    'potential_energy': 'float',
                    'kinetic_energy': 'float',
                    'total_energy': 'float',
                    'pressure': 'float',
                    'volume': 'float',
                    'time': 'float',
                },
            },
            'inputs': {
                'temperature': ['stores', 'temperature'],
                'potential_energy': ['stores', 'potential_energy'],
                'kinetic_energy': ['stores', 'kinetic_energy'],
                'total_energy': ['stores', 'total_energy'],
                'pressure': ['stores', 'pressure'],
                'volume': ['stores', 'volume'],
                'time': ['global_time'],
            },
        },
    }


def register_lammps(core=None):
    """Return a core with LAMMPSProcess, the RAM emitter, and the
    Visualization Step(s) registered."""
    if core is None:
        core = allocate_core()
    core.register_link('LAMMPSProcess', LAMMPSProcess)
    core.register_link('ram-emitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    from pbg_lammps.visualizations import LAMMPSThermoPlots
    core.register_link('LAMMPSThermoPlots', LAMMPSThermoPlots)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_lammps() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_lammps()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
